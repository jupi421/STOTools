import numpy as np
import matplotlib.pyplot as plt

avg = np.zeros((100,4))

for i in range(40, 3690, 50):
    data = np.loadtxt(f"test/op/op{i}.out")
    data_avg = data[:,:4]
    avg += data_avg

avg /= 140

np.savetxt("op_avg.dat", avg)
