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
    plt.legend()
    # plt.show()
    plt.savefig(exp)
        
if __name__ == "__main__":
    logdir1_sb = "data/q1_sb_no_rtg_dsa_CartPole-v0_15-02-2026_11-59-24"
    logdir2_sb = "data/q1_sb_rtg_dsa_CartPole-v0_15-02-2026_12-21-45"
    logdir3_sb = "data/q1_sb_rtg_na_CartPole-v0_15-02-2026_12-23-01"
    
    logdir1_lb = "data/q1_lb_no_rtg_dsa_CartPole-v0_15-02-2026_12-26-29"
    logdir2_lb = "data/q1_lb_rtg_dsa_CartPole-v0_15-02-2026_12-27-35"
    logdir3_lb = "data/q1_lb_rtg_na_CartPole-v0_15-02-2026_12-28-36"
    
    logdirs_sb = [logdir1_sb, logdir2_sb, logdir3_sb]
    logdirs_lb = [logdir1_lb, logdir2_lb, logdir3_lb]
    plot_ret(logdirs_sb, "Small Batch")
    plot_ret(logdirs_lb, "Large Batch")
    
