from trainers import *
from rnn import LeakyRNN
import motornet as mn
import copy, argparse

HIDDEN_UNITS = 64
NUM_STEPS = 8000
BATCH_SIZE = 32
LR = 5e-2


def save_artifact(
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
):
    torch.save(
        {
            "weight_history": weight_history,
            "losses": losses,
            "metrics": metrics,
            "H": H,
            "Y": Y,
            "FT": FT,
            "goal": goal,
            "direction_idx": direction_idx,
            "seed": seed,
            "n_rec": n_rec,
            "num_steps": num_steps,
            "lr": lr,
            "batch_size": batch_size,
            "effector": effector,
        },
        f"results/{effector}_seed{seed}_{rule}.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Experiment Runner",
        description="End-to-end training of RNN for motor task",
    )
    parser.add_argument("effector", choices=["ReluPointMass24", "RigidTendonArm26"])
    args = parser.parse_args()
    print(f"Using {args.effector} effector...")
    reaching_distance = 0.1 if "Arm" in args.effector else 0.5
    effector_cls = getattr(mn.effector, args.effector)
    seeds = range(1, 5)

    for seed in seeds:
        if "Arm" in args.effector:
            effector = effector_cls(mn.muscle.RigidTendonHillMuscle())
        else:
            effector = effector_cls()
        center_out = mn.environment.CenterOutReach(
            effector, reaching_distance=reaching_distance
        )
        bptt_net = LeakyRNN(
            n_in=center_out.observation_space.shape[0],
            n_rec=HIDDEN_UNITS,
            n_out=center_out.n_muscles,
        )
        rflo_net = copy.deepcopy(bptt_net)

        bptttrainer = BPTTTrainer(bptt_net, center_out, position_loss, lr=LR)
        rflotrainer = RFLOTrainer(rflo_net, center_out, position_loss, lr=LR)

        losses, metrics = bptttrainer.train(
            num_steps=NUM_STEPS, batch_size=BATCH_SIZE, seed=seed
        )
        H, Y, FT, goal, direction_idx = bptttrainer.inference(
            options={"direction_idx": np.arange(center_out.n_targets)}, seed=seed
        )
        save_artifact(
            bptttrainer.weight_history,
            losses,
            metrics,
            H,
            Y,
            FT,
            goal,
            direction_idx,
            seed,
            HIDDEN_UNITS,
            NUM_STEPS,
            LR,
            BATCH_SIZE,
            args.effector,
            "BPTT",
        )

        losses, metrics = rflotrainer.train(
            num_steps=NUM_STEPS, batch_size=BATCH_SIZE, seed=seed
        )
        H, Y, FT, goal, direction_idx = rflotrainer.inference(
            options={"direction_idx": np.arange(center_out.n_targets)}, seed=seed
        )

        save_artifact(
            rflotrainer.weight_history,
            losses,
            metrics,
            H,
            Y,
            FT,
            goal,
            direction_idx,
            seed,
            HIDDEN_UNITS,
            NUM_STEPS,
            LR,
            BATCH_SIZE,
            args.effector,
            "RFLO",
        )
