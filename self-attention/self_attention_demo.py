import numpy as np

# taking a input: 3 tokens, each mapped to a 4-dim vector
X = np.random.randn(3, 4)

# learned weight matrices (initialized randomly just for demo)
Wq = np.random.randn(4, 4)
Wk = np.random.randn(4, 4)
Wv = np.random.randn(4, 4)

# compute queries, keys, values
Q = X @ Wq
K = X @ Wk
V = X @ Wv

# attention scores
scores = Q @ K.T / np.sqrt(K.shape[1])

# softmax across last axis
weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

output = weights @ V

print("Input:\n", X)
print("Attention weights:\n", weights)
print("Output:\n", output)
