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
    plt.savefig(f"rep/task_2/{exp}")
        
if __name__ == "__main__":
    logdir = "data/q2_b100_r1e-2_n_80_InvertedPendulum-v4_15-02-2026_13-06-51"
    log = [logdir]
    plot_ret(log, "pendulum")
