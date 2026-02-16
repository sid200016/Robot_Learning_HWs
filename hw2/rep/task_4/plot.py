from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import re


# def short_label(path):
#     match = re.search(r"q4_search_b(\d+)_lr([0-9.]+)_", path)
#     if match:
#         b, lr = match.group(1), match.group(2)
#         return f"b={b}, lr={lr}"
#     return path


def plot_ret(paths, exp):
    plt.figure(figsize=(15, 10))
    T = None
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
        label = "no_rtg_no_baseline"
        if "rtg" in path:
            label = "rtg"
        if "nnbaseline" in path:
            label = "nnbaseline"        
        if "rtg_nnbaseline" in path:
            label = "rtg_nnbaseline"
        
        plt.plot(T, values, label=label)
    plt.xlabel("Time")
    plt.ylabel("Eval_Average_Return")
    plt.legend(fontsize=8)
    plt.title(exp)
 
    # plt.show()
    plt.savefig(f"rep/task_4/{exp}")
        
if __name__ == "__main__":
    # logdir1= "data/q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_15-02-2026_16-30-05"
    # logdir2 = "data/q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_15-02-2026_16-32-14"
    # logdir3 = "data/q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_15-02-2026_16-27-58"
    # logdir4 = "data/q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_15-02-2026_16-40-27"
    # logdir5 = "data/q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_15-02-2026_16-46-36"
    # logdir6 = "data/q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_15-02-2026_16-34-24"
    # logdir7 = "data/q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_15-02-2026_17-02-58"
    # logdir8 = "data/q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_15-02-2026_17-13-07"
    # logdir9 = "data/q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_15-02-2026_16-52-48"
    # log = [logdir1, logdir2, logdir3, logdir4, logdir5, logdir6, logdir7, logdir8, logdir9]
    logdir_opt1 = "data/q4_b_50000_r0.02_rtg_nnbaseline_HalfCheetah-v4_15-02-2026_21-09-04"
    logdir_opt2 = "data/q4_b50000_r0.02_HalfCheetah-v4_15-02-2026_19-25-49"
    logdir_opt3 = "data/q4_b50000_r0.02_nnbaseline_HalfCheetah-v4_15-02-2026_20-35-12"
    logdir_opt4 = "data/q4_b50000_r0.02_rtg_HalfCheetah-v4_15-02-2026_19-39-04"
    log2 = [logdir_opt1, logdir_opt2, logdir_opt3, logdir_opt4]
    plot_ret(log2, "Cheetah_EXP4_Optimal_b_lr")
    # plot_ret(log, "Cheetah_EXP4")

## Optimal -> b = 50000, lr = 0.02