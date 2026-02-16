from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
def plot_ret(paths, exp):
    plt.figure()
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
        plt.plot(T, values, label=path)
    plt.xlabel("Time")
    plt.ylabel("Eval_Average_Return")
    plt.title(exp)
 
    # plt.show()
    plt.savefig(f"rep/task_3/{exp}")
        
if __name__ == "__main__":
    logdir = "data/q3_b10000_r0.005_LunarLanderContinuous-v2_15-02-2026_13-59-54"
    log = [logdir]
    plot_ret(log, "lander")
