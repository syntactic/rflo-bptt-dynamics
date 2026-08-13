from trainers import *
from rnn import LeakyRNN
import motornet as mn
import copy

HIDDEN_UNITS = 64
NUM_STEPS = 8000
BATCH_SIZE = 32
LR = 1e-3

if __name__ == "__main__":

    seeds = range(3, 5)

    for seed in seeds:
        relu_point_mass = mn.effector.ReluPointMass24()
        center_out = mn.environment.CenterOutReach(relu_point_mass, reaching_distance=0.5)
        bptt_net = LeakyRNN(n_in=center_out.observation_space.shape[0], n_rec=HIDDEN_UNITS, n_out=center_out.n_muscles)
        rflo_net = copy.deepcopy(bptt_net)

        bptttrainer = BPTTTrainer(bptt_net, center_out, position_loss, lr=LR)
        rflotrainer = RFLOTrainer(rflo_net, center_out, position_loss, lr=LR)

        losses, metrics = bptttrainer.train(num_steps=NUM_STEPS, batch_size=BATCH_SIZE, seed=seed)
        H, Y, FT, goal, direction_idx = bptttrainer.inference(options={"direction_idx":np.arange(center_out.n_targets)}, seed=seed)
        torch.save({"weight_history": bptttrainer.weight_history, "losses": losses, "metrics": metrics, "H":H, "Y": Y, "FT": FT,
                    "goal": goal, "direction_idx": direction_idx, "seed":seed, "n_rec": HIDDEN_UNITS, "num_steps": NUM_STEPS,
                    "lr": LR, "batch_size": BATCH_SIZE}, f"results/seed{seed}_BPTT.pt")

        losses, metrics = rflotrainer.train(num_steps=NUM_STEPS, batch_size=BATCH_SIZE, seed=seed)
        H, Y, FT, goal, direction_idx = rflotrainer.inference(options={"direction_idx":np.arange(center_out.n_targets)}, seed=seed)
        torch.save({"weight_history": rflotrainer.weight_history, "losses": losses, "metrics": metrics, "H":H, "Y": Y, "FT": FT,
                    "goal": goal, "direction_idx": direction_idx, "seed":seed, "n_rec": HIDDEN_UNITS, "num_steps": NUM_STEPS,
                    "lr": LR, "batch_size": BATCH_SIZE}, f"results/seed{seed}_RFLO.pt")
