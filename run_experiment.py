import argparse
import copy
from pathlib import Path

import motornet as mn
import numpy as np
import torch

from rnn import LeakyRNN
from trainers import *


def save_artifact(
    output_dir,
    weight_history,
    losses,
    metrics,
    H,
    Y,
    FT,
    goal,
    direction_idx,
    seed,
    n_rec,
    num_steps,
    lr,
    batch_size,
    effector,
    rule,
    proprioception_noise,
    vision_noise,
    b_seed,
    loss_meta,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{effector}_seed{seed}_{rule}.pt"

    def _to_cpu(x):
        if torch.is_tensor(x):
            return x.detach().cpu()
        return x

    torch.save(
        {
            "weight_history": (
                [_to_cpu(w) for w in weight_history]
                if isinstance(weight_history, list)
                else _to_cpu(weight_history)
            ),
            "losses": losses,
            "metrics": metrics,
            "H": _to_cpu(H),
            "Y": _to_cpu(Y),
            "FT": _to_cpu(FT),
            "goal": _to_cpu(goal),
            "direction_idx": _to_cpu(direction_idx),
            "seed": seed,
            "n_rec": n_rec,
            "num_steps": num_steps,
            "lr": lr,
            "batch_size": batch_size,
            "effector": effector,
            # Provenance for the clean-vs-noisy condition. Zero on clean runs;
            # b_seed is meaningful only for the RFLO feedback matrix (None for BPTT).
            "proprioception_noise": proprioception_noise,
            "vision_noise": vision_noise,
            "b_seed": b_seed,
            # {"type": "trajectory"} or {"type": "terminal", "w_term":..., "k":...}
            "loss": loss_meta,
        },
        file_path,
    )
    print(f"Saved artifact to {file_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Experiment Runner",
        description="End-to-end training of LeakyRNN on MotorNet center-out task comparing BPTT vs RFLO",
    )
    parser.add_argument(
        "effector",
        choices=["ReluPointMass24", "RigidTendonArm26"],
        help="MotorNet effector model",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of seeds to run (e.g. --seeds 0 1 2 3 4)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seeds to run if --seeds is omitted (0 to n-1, default: 5)",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=64,
        help="Number of recurrent units (default: 64)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=8000,
        help="Training iterations per model (default: 8000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-2, help="Learning rate for SGD (default: 0.05)"
    )
    parser.add_argument(
        "--loss",
        choices=["trajectory", "terminal"],
        default="trajectory",
        help="'trajectory' = time-averaged L1 (default); 'terminal' weights the last --terminal-k steps by --terminal-weight",
    )
    parser.add_argument(
        "--terminal-weight",
        type=float,
        default=0.5,
        help="Weight on terminal error for --loss terminal (default: 0.5)",
    )
    parser.add_argument(
        "--terminal-k",
        type=int,
        default=10,
        help="Number of trailing timesteps in the terminal loss term (default: 10)",
    )
    parser.add_argument(
        "--rule",
        choices=["both", "bptt", "rflo"],
        default="both",
        help="Which learning rule(s) to train per seed (default: 'both'). "
        "'rflo' skips BPTT training; both nets are still built so the shared "
        "initialization and RNG stream are identical to a 'both' run.",
    )
    parser.add_argument(
        "--b-seed",
        type=str,
        default="0",
        help="Seed for RFLO random feedback matrix B: integer (e.g. '0' for fixed across runs) or 'match' to match the network seed (default: '0')",
    )
    parser.add_argument(
        "--proprioception-noise",
        type=float,
        default=0.0,
        help="Std of Gaussian proprioceptive-feedback noise (default: 0.0 = clean). Scale per-effector.",
    )
    parser.add_argument(
        "--vision-noise",
        type=float,
        default=0.0,
        help="Std of Gaussian visual-feedback noise, in the effector's position units (default: 0.0 = clean). Scale per-effector.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save artifacts (default: 'results'). Use e.g. 'results/clean' or 'results/noisy' to separate conditions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on ('cpu', 'cuda', 'mps', etc., default: cuda if available else cpu)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))

    print(
        f"Running experiment on {args.effector} across seeds {seeds} (num_steps={args.num_steps}, lr={args.lr}, b_seed={args.b_seed}, device={args.device})..."
    )
    reaching_distance = 0.1 if "Arm" in args.effector else 0.5
    effector_cls = getattr(mn.effector, args.effector)

    # Same loss object drives both trainers so BPTT vs RFLO stays matched.
    if args.loss == "terminal":
        loss_fn = make_terminal_weighted_loss(args.terminal_weight, args.terminal_k)
        loss_meta = {
            "type": "terminal",
            "w_term": args.terminal_weight,
            "k": args.terminal_k,
        }
    else:
        loss_fn = position_loss
        loss_meta = {"type": "trajectory"}

    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---")
        # Build fresh effector and environment per seed to prevent tensor autograd graph retention in deepcopy
        if "Arm" in args.effector:
            effector = effector_cls(mn.muscle.RigidTendonHillMuscle())
        else:
            effector = effector_cls()
        center_out = mn.environment.CenterOutReach(
            effector,
            reaching_distance=reaching_distance,
            proprioception_noise=args.proprioception_noise,
            vision_noise=args.vision_noise,
        )

        torch.manual_seed(seed)
        bptt_net = LeakyRNN(
            n_in=center_out.observation_space.shape[0],
            n_rec=args.hidden_units,
            n_out=center_out.n_muscles,
        )
        rflo_net = copy.deepcopy(bptt_net)

        # Resolve RFLO feedback matrix B seed
        rflo_b_seed = seed if args.b_seed.lower() == "match" else int(args.b_seed)

        bptttrainer = BPTTTrainer(
            bptt_net, center_out, loss_fn, lr=args.lr, device=args.device
        )
        rflotrainer = RFLOTrainer(
            rflo_net,
            center_out,
            loss_fn,
            lr=args.lr,
            seed=rflo_b_seed,
            device=args.device,
        )

        if args.rule in ("both", "bptt"):
            print(f"Training BPTT (seed={seed})...")
            losses, metrics = bptttrainer.train(
                num_steps=args.num_steps, batch_size=args.batch_size, seed=seed
            )
            H, Y, FT, goal, direction_idx = bptttrainer.inference(
                options={"direction_idx": np.arange(center_out.n_targets)}, seed=seed
            )
            save_artifact(
                args.output_dir,
                bptttrainer.weight_history,
                losses,
                metrics,
                H,
                Y,
                FT,
                goal,
                direction_idx,
                seed,
                args.hidden_units,
                args.num_steps,
                args.lr,
                args.batch_size,
                args.effector,
                "BPTT",
                args.proprioception_noise,
                args.vision_noise,
                None,  # B is not used by BPTT
                loss_meta,
            )

        if args.rule in ("both", "rflo"):
            print(f"Training RFLO (seed={seed}, b_seed={rflo_b_seed})...")
            losses, metrics = rflotrainer.train(
                num_steps=args.num_steps, batch_size=args.batch_size, seed=seed
            )
            H, Y, FT, goal, direction_idx = rflotrainer.inference(
                options={"direction_idx": np.arange(center_out.n_targets)}, seed=seed
            )
            save_artifact(
                args.output_dir,
                rflotrainer.weight_history,
                losses,
                metrics,
                H,
                Y,
                FT,
                goal,
                direction_idx,
                seed,
                args.hidden_units,
                args.num_steps,
                args.lr,
                args.batch_size,
                args.effector,
                "RFLO",
                args.proprioception_noise,
                args.vision_noise,
                rflo_b_seed,
                loss_meta,
            )
