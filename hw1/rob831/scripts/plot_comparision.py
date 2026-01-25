import numpy as np
import matplotlib.pyplot as plt

train_batch_sizes = [1, 64, 128, 256, 512, 1024, 2000]
BC_Average = [882.6964, 828.8473, 706.3738, 1408.4910, 3564.03174, 2031.8616, 3653.4609]
BC_std = [5.3870, 137.6664, 80.7276, 1147.46801, 1398.3375, 1492.1425, 1432.5129]

plt.figure()
plt.title('Behavior Cloning Performance vs Train Batch Size for 10 eval rollouts')
plt.plot(train_batch_sizes, BC_Average, label='BC Eval Average Return')
plt.plot(train_batch_sizes, BC_std, label='BC Eval Std Return')
plt.xlabel('Train Batch Size')
plt.ylabel('Return Mean and Std')
plt.legend()
plt.savefig('bc_comparison.png')