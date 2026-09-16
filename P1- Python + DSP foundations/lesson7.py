import numpy as np
import matplotlib.pyplot as plt

fs = 1000          # samples/second
f = 50             # signal frequency
duration = 0.1

t = np.arange(0, duration, 1/fs)

I = np.cos(2 * np.pi * f * t)
Q = np.sin(2 * np.pi * f * t)

iq = I + 1j * Q

plt.plot(t, I, label="I")
plt.plot(t, Q, label="Q")

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("I/Q Components")
plt.legend()
plt.grid()
plt.show()