import numpy as np
import matplotlib.pyplot as plt

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))

    # Initialize positional encoding matrix with zeros
    for k in range(seq_len):
        # Loop over each dimension of the positional encoding
        for i in np.arange(int(d/2)):
            # Calculate the denominator for the sinusoidal function
            denominator = np.power(n, (2*i)/d)
            # Compute sine and cosine components of the positional encoding
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

P = getPositionEncoding(seq_len=4, d=4, n=100)
print(P)

