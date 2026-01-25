import numpy as np
import pickle

def analyze_expert_data(data):
    traj1 = data[0]
    traj2 = data[1]
    return_1 = np.sum(traj1['reward'])
    return_2 = np.sum(traj2['reward'])
    mean_return = np.mean([return_1, return_2])
    std_return = np.std([return_1, return_2])
    return mean_return, std_return


if __name__ == "__main__":
    data_antv2 = pickle.load(open("rob831/expert_data/expert_data_Ant-v2.pkl", "rb"))
    data_halfcheetahv2 = pickle.load(open("rob831/expert_data/expert_data_HalfCheetah-v2.pkl", "rb"))
    data_hopperv2 = pickle.load(open("rob831/expert_data/expert_data_Hopper-v2.pkl", "rb"))
    data_humanoidv2 = pickle.load(open("rob831/expert_data/expert_data_Humanoid-v2.pkl", "rb"))
    data_walkerv2 = pickle.load(open("rob831/expert_data/expert_data_Walker2d-v2.pkl", "rb"))
    print(len(data_antv2[0]['reward']))
    print("Ant-v2 mean and std of returns:", analyze_expert_data(data_antv2))
    print("Humanoid-v2 mean and std of returns:", analyze_expert_data(data_humanoidv2))
    print("Walker-v2 mean and std of returns:", analyze_expert_data(data_walkerv2))
    print("Hopper-v2 mean and std of returns:", analyze_expert_data(data_hopperv2))
    print("HalfCheetah-v2 mean and std of returns:", analyze_expert_data(data_halfcheetahv2))
    

    

##Ant-v2 dict_keys(['observation', 'image_obs', 'reward', 'action', 'next_observation', 'terminal'])
