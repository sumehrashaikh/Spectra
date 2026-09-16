import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Number of bits
num_bits = 100

# Random bits: 0 or 1
bits = np.random.randint(0, 2, num_bits)

# BPSK mapping
symbols = 2 * bits - 1

# Plot constellation
plt.scatter(
    symbols,
    np.zeros_like(symbols),
    alpha=0.7
)

plt.xlabel("I")
plt.ylabel("Q")
plt.title("BPSK Constellation")

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

plt.grid()
plt.show()

print("First 20 bits:")
print(bits[:20])

print("First 20 symbols:")
print(symbols[:20])