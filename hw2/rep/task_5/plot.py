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

        
        plt.plot(T, values, label=short_label(path))
    plt.xlabel("Time")
    plt.ylabel("Eval_Average_Return")
    plt.legend(fontsize=8)
    plt.title(exp)
 
    # plt.show()
    plt.savefig(f"rep/task_5/{exp}")
        
if __name__ == "__main__":
    logdir1 = "data/q5_b2000_r0.001_lambda0_Hopper-v4_16-02-2026_00-21-53"
    logdir2 = "data/q5_b2000_r0.001_lambda0.95_Hopper-v4_16-02-2026_00-24-03"
    logdir3 = "data/q5_b2000_r0.001_lambda0.99_Hopper-v4_16-02-2026_00-26-24"
    logdir4 = "data/q5_b2000_r0.001_lambda1_Hopper-v4_16-02-2026_00-28-44"
    log = [logdir1, logdir2, logdir3, logdir4]
    plot_ret(log, "EXP5_lambda_GAE")
    # plot_ret(log, "Cheetah_EXP4")

## Optimal -> b = 50000, lr = 0.02