from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import re
def short_label(path):
    match = re.search(r"q4_search_b(\d+)_lr([0-9.]+)_", path)
    if match:
        b, lr = match.group(1), match.group(2)
        return f"b={b}, lr={lr}"
    return path



def plot_ret(paths, exp, leg_labels):
    plt.figure(figsize=(15, 10))
    T = None
    i = 0
    for path in paths: 
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()

        tag = "Eval_AverageReturn"
        events = ea.Scalars(tag)
        values = np.array([e.value for e in events], dtype=np.float32)
        if T is None:
            time_tag = "TimeSinceStart"
            event_T = ea.Scalars(time_tag)
            T = np.array([e.value for e in event_T], dtype=np.float32)

        
        plt.plot(T, values, label=f"num sequences: {leg_labels[i]}")
        i += 1
    plt.xlabel("Time")
    plt.ylabel("Eval_Average_Return")
    plt.legend(fontsize=8)
    plt.title(exp)
 
    # plt.show()
    plt.savefig(f"rep/{exp}")

if __name__ == "__main__":
    # ens1 = "rob831/data/hw4_q4_reacher_ensemble1_reacher-hw4_part1-v0_26-03-2026_12-45-11"
    # ens2 = "rob831/data/hw4_q4_reacher_ensemble3_reacher-hw4_part1-v0_26-03-2026_12-47-40"
    # ens3 = "rob831/data/hw4_q4_reacher_ensemble5_reacher-hw4_part1-v0_26-03-2026_12-54-23"
    
    # plot_ret([ens1, ens2, ens3], "ensemble", [1, 3, 5])
    
    
    # hor1 = "rob831/data/hw4_q4_reacher_horizon5_reacher-hw4_part1-v0_26-03-2026_09-47-38"
    # hor2 = "rob831/data/hw4_q4_reacher_horizon15_reacher-hw4_part1-v0_26-03-2026_09-52-56"
    # hor3 = "rob831/data/hw4_q4_reacher_horizon30_reacher-hw4_part1-v0_26-03-2026_11-17-16"
    
    # plot_ret([hor1, hor2, hor3], "horizon", [5, 15, 30])
    
    num_seq1 = "rob831/data/hw4_q4_reacher_numseq100_reacher-hw4_part1-v0_26-03-2026_12-35-46"
    num_seq2 = "rob831/data/hw4_q4_reacher_numseq1000_reacher-hw4_part1-v0_26-03-2026_12-38-29"
    
    plot_ret([num_seq1, num_seq2], "num_sequencues", [100, 1000])