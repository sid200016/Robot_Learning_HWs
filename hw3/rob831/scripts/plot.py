import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def short_label(path):
    match = re.search(r"q4_search_b(\d+)_lr([0-9.]+)_", path)
    if match:
        b, lr = match.group(1), match.group(2)
        return f"b={b}, lr={lr}"
    return path


def get_section_results(file):
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == "Train_EnvstepsSoFar":
                X.append(v.simple_value)
            elif v.tag == "Train_AverageReturn":
                Y.append(v.simple_value)
        
    return X, Y


def plot_ret(paths, exp, err=False):
    plt.figure(figsize=(15, 10))
    T = None
    all_values = []
    for path in paths:
        logdir = os.path.join(path, "events*")
        eventfile = glob.glob(logdir)[0]
        steps, returns = get_section_results(eventfile)

        if T is None:
            T = np.array(steps, dtype=np.float32)
        values = np.array(returns, dtype=np.float32)
        all_values.append(values)
        


   
    plt.xlabel("Time")
    plt.ylabel("Train_Average_Return")

    plt.title(exp)
    if err:
        
        all_values = np.stack(all_values, axis = 0)
        mean = np.mean(all_values, axis = 0)
        stddev = np.std(all_values, axis = 0)
        if len(T) != len(mean):
            T = T[:-1]
        plt.errorbar(T, mean, stddev, fmt = '-o', capsize=3)
    else:
        plt.plot(T, all_values[0])
    # plt.show()
    
    plt.savefig(f"figs/{exp}")
        
if __name__ == "__main__":
    lander_dq1 = "data/q1_dqn_1_LunarLander-v3_24-02-2026_10-21-10"
    lander_dq2 = "data/q1_dqn_2_LunarLander-v3_23-02-2026_15-01-47"
    lander_dq3 = "data/q1_dqn_3_LunarLander-v3_23-02-2026_15-06-50"
    
    lander_dq = [lander_dq1, lander_dq2, lander_dq3]
    plot_ret(lander_dq, "LunarLander_DQN", True)
    
    lander_ddq1 = "data/q1_doubledqn_1_LunarLander-v3_23-02-2026_15-13-32"
    lander_ddq2 = "data/q1_doubledqn_2_LunarLander-v3_24-02-2026_10-27-02"
    lander_ddq3 = "data/q1_doubledqn_3_LunarLander-v3_23-02-2026_15-34-32"
    lander_ddq = [lander_ddq1, lander_ddq2, lander_ddq3]
    plot_ret(lander_ddq, "LunarLander_DDQN", True)
    
    cartpole = ["data/q2_10_10_CartPole-v0_24-02-2026_11-23-34"]
    pend = ["data/q3_10_10_InvertedPendulum-v4_24-02-2026_11-27-01"]
    
    plot_ret(cartpole, "cartpole")
    plot_ret(pend, "inverted pendulum")
