import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("output.csv", delimiter=",")
plt.scatter(-data[0, :], data[1, :], cmap="rainbow", c=np.arange(data.shape[1]))
plt.savefig("output.png")
