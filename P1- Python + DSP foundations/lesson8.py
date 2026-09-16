import numpy as np
import matplotlib.pyplot as plt

fs = 1000
f = 50
duration = 0.1

t = np.arange(0, duration, 1/fs)

I = np.cos(2 * np.pi * f * t)
Q = np.sin(2 * np.pi * f * t)

plt.plot(I, Q)

plt.xlabel("I")
plt.ylabel("Q")
plt.title("I/Q Plane")
plt.axis("equal")
plt.grid()
plt.show()