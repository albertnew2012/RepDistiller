"""
checkpoint_drift.py
===================
Measure how much each layer ACTUALLY learned between two checkpoints.

Unlike visualize_grad_flow.py (which inspects a single backward pass -- i.e.
"how much is each layer about to MOVE this step"), this tool measures the
CUMULATIVE change in the weights between two saved checkpoints -- i.e. "how much
did each layer actually move over all the training in between".

For each top-level stage it reports the **relative drift**:

    relative_drift(stage) = || W_after - W_before || / || W_before ||

This is the honest "did this layer learn?" signal discussed in GRADIENT_FLOW.md:
a single-step gradient cannot tell you this; only comparing two real checkpoints
can. Pair it with the validation accuracy stored in each checkpoint to connect
"weights moved" with "model got better".

Run it:
    cd ~/Desktop/RepDistiller
    CKDIR="save/student_model/S:resnet8x4_T:resnet32x4_cifar100_kd_r:0.1_a:0.9_b:0.0_debug"
    python3 checkpoint_drift.py --before "$CKDIR/ckpt_epoch_40.pth" \
                                --after  "$CKDIR/ckpt_epoch_80.pth"
"""

from __future__ import print_function

import argparse

import torch


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
def parse_args():
    base = ("save/student_model/"
            "S:resnet8x4_T:resnet32x4_cifar100_kd_r:0.1_a:0.9_b:0.0_debug")
    p = argparse.ArgumentParser(description="Per-layer weight drift between two checkpoints")
    p.add_argument("--before", type=str,
                   default=base + "/ckpt_epoch_40.pth",
                   help="earlier checkpoint")
    p.add_argument("--after", type=str,
                   default=base + "/ckpt_epoch_80.pth",
                   help="later checkpoint")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  ASCII bar chart helper
# --------------------------------------------------------------------------- #
def ascii_bars(title, labels, values, width=48):
    print("\n" + title)
    print("-" * len(title))
    vmax = max(values) if values and max(values) > 0 else 1.0
    for lbl, v in zip(labels, values):
        n = int(round(width * v / vmax))
        print("  {:<8} | {:<{w}} {:.4e}".format(lbl, "#" * n, v, w=width))


def load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["model"]
    acc = ck.get("best_acc", ck.get("accuracy"))
    epoch = ck.get("epoch")
    return state, acc, epoch


def stage_of(param_name):
    """Bucket a parameter into its top-level stage (conv1/layer1/.../fc)."""
    return param_name.split(".")[0]


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    opt = parse_args()

    state_b, acc_b, ep_b = load_ckpt(opt.before)
    state_a, acc_a, ep_a = load_ckpt(opt.after)

    print("==> BEFORE: {}  (epoch={}, acc={})".format(opt.before, ep_b, acc_b))
    print("==> AFTER : {}  (epoch={}, acc={})".format(opt.after, ep_a, acc_a))

    stage_order = ["conv1", "layer1", "layer2", "layer3", "fc"]
    diff_sq = {s: 0.0 for s in stage_order}   # sum ||W_after - W_before||^2
    base_sq = {s: 0.0 for s in stage_order}   # sum ||W_before||^2

    skipped = 0
    for name, w_b in state_b.items():
        s = stage_of(name)
        if s not in diff_sq:
            continue
        if name not in state_a:
            skipped += 1
            continue
        # Only compare float tensors (skip BN integer buffers like
        # num_batches_tracked, which are counters, not learned weights).
        if not torch.is_floating_point(w_b):
            continue
        w_a = state_a[name]
        diff_sq[s] += (w_a - w_b).norm().item() ** 2
        base_sq[s] += w_b.norm().item() ** 2

    eps = 1e-12
    abs_drift = [diff_sq[s] ** 0.5 for s in stage_order]
    rel_drift = [(diff_sq[s] ** 0.5) / (base_sq[s] ** 0.5 + eps) for s in stage_order]

    ascii_bars("ABSOLUTE DRIFT   ||W_after - W_before||   (raw amount moved)",
               stage_order, abs_drift)

    ascii_bars("RELATIVE DRIFT   ||W_after - W_before|| / ||W_before||   (fair: did it learn?)",
               stage_order, rel_drift)
    print("  This is the cumulative version of the relative-update panel:")
    print("  it shows how much each stage ACTUALLY changed between the two checkpoints.")

    if acc_b is not None and acc_a is not None:
        try:
            da = float(acc_a) - float(acc_b)
            print("\n==> validation accuracy: {:.2f}% -> {:.2f}%   (delta {:+.2f}%)"
                  .format(float(acc_b), float(acc_a), da))
            print("    'weights moved' (above) + 'accuracy went up' (here) = learning.")
        except (TypeError, ValueError):
            pass

    if skipped:
        print("\n==> note: {} parameters present in BEFORE were missing in AFTER (skipped)."
              .format(skipped))


if __name__ == "__main__":
    main()
