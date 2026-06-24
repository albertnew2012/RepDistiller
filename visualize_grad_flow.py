"""
visualize_grad_flow.py
======================
An educational, standalone script to *see* how the KD loss produces gradients
and how those gradients flow backward through the student network.

It mirrors exactly what helper/loops.py::train_distill does for `--distill kd`,
but on a SINGLE batch, with instrumentation:

  1. Builds the real teacher (resnet32x4) and student (resnet8x4).
  2. Runs one forward pass and computes the same loss:
         loss = gamma*loss_cls + alpha*loss_div + beta*loss_kd
  3. Registers hooks on every student layer to capture the gradient that
     flows *out the back* of that layer during loss.backward().
  4. Prints:
       - the three loss terms and their weighted contributions
       - a per-layer table of gradient magnitudes (the "gradient signal")
       - an ASCII bar chart so you can see the shape of the flow
  5. Saves a matplotlib figure (grad_flow.png) with two panels:
       - gradient norm of the activations at each layer (the signal travelling
         backward through the graph)
       - gradient norm of the weights at each layer (what actually updates)

Run it:
    cd ~/Desktop/RepDistiller
    python3 visualize_grad_flow.py
    python3 visualize_grad_flow.py --distill crd        # compare a feature method
    python3 visualize_grad_flow.py -r 1.0 -a 0.0        # only hard labels, no teacher
"""

from __future__ import print_function

import argparse

# NOTE: import torch (and warm Triton via torch._dynamo) BEFORE anything that
# may pull in TensorFlow, to avoid the Triton/TF import-order segfault.
import torch
import torch._dynamo  # noqa: F401  (only imported to initialise Triton early)
import torch.nn as nn

from models import model_dict
from distiller_zoo import DistillKL


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Visualize KD gradient flow")
    p.add_argument("--path_t", type=str,
                   default="./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth",
                   help="teacher checkpoint")
    p.add_argument("--model_s", type=str, default="resnet8x4", help="student arch")
    p.add_argument("--path_s", type=str, default=None,
                   help="optional student checkpoint to load (e.g. a trained "
                        "resnet8x4_best.pth). If omitted, the student is randomly "
                        "initialised.")
    p.add_argument("--distill", type=str, default="kd",
                   choices=["kd"], help="only kd is wired up in this demo")
    p.add_argument("-r", "--gamma", type=float, default=0.1,
                   help="weight for classification (hard labels)")
    p.add_argument("-a", "--alpha", type=float, default=0.9,
                   help="weight for KL divergence (teacher soft labels)")
    p.add_argument("-b", "--beta", type=float, default=0.0,
                   help="weight for extra feature loss (0 for plain kd)")
    p.add_argument("--kd_T", type=float, default=4.0, help="KD temperature")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05,
                   help="learning rate used for the relative-update panel "
                        "(matches train_student.py default of 0.05)")
    p.add_argument("--no_plot", action="store_true", help="skip saving the PNG")
    return p.parse_args()


def teacher_name_from_path(path):
    return path.split("/")[-2].split("_")[0]


# --------------------------------------------------------------------------- #
#  Build models exactly like train_student.py
# --------------------------------------------------------------------------- #
def build_models(opt, n_cls=100):
    t_name = teacher_name_from_path(opt.path_t)
    print("==> teacher: {}   student: {}".format(t_name, opt.model_s))

    model_t = model_dict[t_name](num_classes=n_cls)
    state = torch.load(opt.path_t, map_location="cpu", weights_only=False)
    model_t.load_state_dict(state["model"])

    model_s = model_dict[opt.model_s](num_classes=n_cls)
    if opt.path_s is not None:
        # Load a real (trained) student so the gradients reflect a model that
        # has actually learned, instead of random initialisation.
        s_state = torch.load(opt.path_s, map_location="cpu", weights_only=False)
        model_s.load_state_dict(s_state["model"])
        acc = s_state.get("best_acc", s_state.get("accuracy"))
        print("==> loaded trained student from {}  (acc={})".format(opt.path_s, acc))
    else:
        print("==> student is RANDOMLY initialised (no --path_s given)")
    return model_t, model_s


# --------------------------------------------------------------------------- #
#  Hooks: capture the gradient flowing OUT the back of each named layer
# --------------------------------------------------------------------------- #
def register_activation_grad_hooks(model_s):
    """
    We attach a full backward hook on each of the student's top-level stages.
    grad_output is the gradient w.r.t. that module's OUTPUT activations -- i.e.
    the error signal arriving from the layers above during backprop. Its norm
    tells us "how strong is the learning signal at this depth".
    """
    captured = {}  # name -> grad-output L2 norm

    # The interesting, ordered stages of a CIFAR ResNet student.
    named_stages = [
        ("conv1", model_s.conv1),
        ("layer1", model_s.layer1),
        ("layer2", model_s.layer2),
        ("layer3", model_s.layer3),
        ("fc", model_s.fc),
    ]

    handles = []
    for name, module in named_stages:
        def make_hook(layer_name):
            def hook(mod, grad_input, grad_output):
                # grad_output is a tuple; take the first (main) tensor.
                g = grad_output[0]
                if g is not None:
                    captured[layer_name] = g.detach().norm().item()
            return hook
        handles.append(module.register_full_backward_hook(make_hook(name)))

    order = [n for n, _ in named_stages]
    return captured, order, handles


# --------------------------------------------------------------------------- #
#  ASCII bar chart helper
# --------------------------------------------------------------------------- #
def ascii_bars(title, labels, values, width=48):
    print("\n" + title)
    print("-" * (len(title)))
    vmax = max(values) if values and max(values) > 0 else 1.0
    for lbl, v in zip(labels, values):
        n = int(round(width * v / vmax))
        bar = "#" * n
        print("  {:<8} | {:<{w}} {:.4e}".format(lbl, bar, v, w=width))


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    opt = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==> device:", device)

    model_t, model_s = build_models(opt)
    model_t.to(device).eval()      # teacher: frozen, eval mode
    model_s.to(device).train()     # student: learning, train mode

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    # ---- one synthetic batch (random images + random labels) -------------- #
    # Random data is fine: we only want to observe the gradient mechanics,
    # not actually train. Shapes match CIFAR (3x32x32, 100 classes).
    torch.manual_seed(0)
    x = torch.randn(opt.batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 100, (opt.batch_size,), device=device)

    # ---- instrument the student ------------------------------------------ #
    act_grads, stage_order, handles = register_activation_grad_hooks(model_s)

    # ===================== forward (same as train_distill) ================= #
    feat_s, logit_s = model_s(x, is_feat=True)
    with torch.no_grad():
        feat_t, logit_t = model_t(x, is_feat=True)
        logit_t = logit_t.detach()      # teacher logits are a CONSTANT target

    loss_cls = criterion_cls(logit_s, y)            # hard-label CE
    loss_div = criterion_div(logit_s, logit_t)      # soft-label KL
    loss_kd = torch.tensor(0.0, device=device)      # 0 for plain kd

    loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

    # ---- report the loss decomposition ----------------------------------- #
    print("\n================= LOSS DECOMPOSITION =================")
    print("  loss_cls (CE  vs true labels) = {:.4f}   x gamma {:.2f} = {:.4f}"
          .format(loss_cls.item(), opt.gamma, opt.gamma * loss_cls.item()))
    print("  loss_div (KL  vs teacher    ) = {:.4f}   x alpha {:.2f} = {:.4f}"
          .format(loss_div.item(), opt.alpha, opt.alpha * loss_div.item()))
    print("  loss_kd  (feature term      ) = {:.4f}   x beta  {:.2f} = {:.4f}"
          .format(loss_kd.item(), opt.beta, opt.beta * loss_kd.item()))
    print("  ----------------------------------------------------")
    print("  TOTAL loss                    = {:.4f}".format(loss.item()))

    # ===================== backward ======================================= #
    model_s.zero_grad()
    loss.backward()     # <-- this fills .grad on every student weight,
    #                         and fires our hooks layer by layer.

    # ---- 1) activation-gradient flow (the signal travelling backward) ---- #
    act_vals = [act_grads.get(n, 0.0) for n in stage_order]
    ascii_bars("ACTIVATION-GRADIENT NORM  (error signal arriving at each stage)",
               stage_order, act_vals)
    print("  Read backward (fc -> conv1): this is the path loss.backward() takes.")

    # ---- 2) weight-gradient norms (what the optimizer will actually use) -- #
    weight_stage_norms = {n: 0.0 for n in stage_order}
    for pname, p in model_s.named_parameters():
        if p.grad is None:
            continue
        # bucket each parameter into its top-level stage
        bucket = pname.split(".")[0]
        if bucket in weight_stage_norms:
            weight_stage_norms[bucket] += p.grad.detach().norm().item() ** 2
    weight_vals = [weight_stage_norms[n] ** 0.5 for n in stage_order]
    ascii_bars("WEIGHT-GRADIENT NORM      (magnitude of update per stage)",
               stage_order, weight_vals)
    print("  This is dLoss/dW: the optimizer.step() moves each weight against it.")

    # ---- 3) RELATIVE update per stage: ||lr*grad|| / ||W|| ---------------- #
    # This is the honest "how much did this layer actually move?" metric.
    # It divides the update size by how big the weights already are, removing
    # the parameter-count / weight-scale bias that inflates the raw norm.
    update_sq = {n: 0.0 for n in stage_order}   # sum of (lr*grad)^2
    weight_sq = {n: 0.0 for n in stage_order}   # sum of (W)^2
    for pname, p in model_s.named_parameters():
        bucket = pname.split(".")[0]
        if bucket not in update_sq:
            continue
        weight_sq[bucket] += p.detach().norm().item() ** 2
        if p.grad is not None:
            update_sq[bucket] += (opt.lr * p.grad.detach().norm().item()) ** 2
    eps = 1e-12
    rel_vals = [(update_sq[n] ** 0.5) / (weight_sq[n] ** 0.5 + eps)
                for n in stage_order]
    ascii_bars("RELATIVE UPDATE  ||lr*grad|| / ||W||   (how much each stage moves)",
               stage_order, rel_vals)
    print("  lr={}. This normalises out parameter count & weight scale --".format(opt.lr))
    print("  it is the fair 'did this layer actually learn this step?' metric.")

    # ---- teacher sanity check: it must have NO gradients ----------------- #
    t_has_grad = any(p.grad is not None for p in model_t.parameters())
    print("\n==> teacher received gradients? {}  (expected: False -- it is frozen)"
          .format(t_has_grad))

    for h in handles:
        h.remove()

    # ---- 3) optional matplotlib figure ----------------------------------- #
    if not opt.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(17, 4))
            axes[0].bar(stage_order, act_vals, color="#4C72B0")
            axes[0].set_title("Activation-gradient norm\n(error signal at each stage)")
            axes[0].set_ylabel("||grad of activations||")
            axes[1].bar(stage_order, weight_vals, color="#C44E52")
            axes[1].set_title("Weight-gradient norm\n(size of the update)")
            axes[1].set_ylabel("||dLoss/dW||")
            axes[2].bar(stage_order, rel_vals, color="#55A868")
            axes[2].set_title("Relative update\n||lr*grad|| / ||W||")
            axes[2].set_ylabel("fraction of weight magnitude")
            for ax in axes:
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle(
                "KD gradient flow  (gamma={}, alpha={}, beta={}, T={})"
                .format(opt.gamma, opt.alpha, opt.beta, opt.kd_T))
            fig.tight_layout()
            out = "grad_flow.png"
            fig.savefig(out, dpi=120)
            print("\n==> saved figure to {}".format(out))
        except Exception as e:  # pragma: no cover - plotting is optional
            print("\n==> (matplotlib unavailable, skipped PNG: {})".format(e))


if __name__ == "__main__":
    main()
