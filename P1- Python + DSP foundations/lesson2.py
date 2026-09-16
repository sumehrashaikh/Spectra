import numpy as np
import matplotlib.pyplot as plt

fs = 60
f = 50
duration = 0.2

t = np.arange(0, duration, 1/fs)
x = np.sin(2 * np.pi * f * t)

plt.plot(t, x, 'o-')
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Aliasing Example")
plt.grid()
plt.show()