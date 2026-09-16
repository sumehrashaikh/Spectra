import numpy as np

iq = np.array([
    1 + 2j,
    3 + 4j,
    5 + 6j,
    7 + 8j
])

print("IQ samples:", iq)

print("I:", iq.real)
print("Q:", iq.imag)

print("Magnitude:", np.abs(iq))