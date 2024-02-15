# Copyright © 2023 Apple Inc.

import time
import mlx.core as mx

num_features = 100
num_examples = 1_000
num_iters = 10_000
lr = 0.01

# True parameters
w_star = mx.random.normal((num_features,))

# Input examples (design matrix)
X = mx.random.normal((num_examples, num_features))

# Noisy labels
eps = 1e-2 * mx.random.normal((num_examples,))
y = X @ w_star + eps

# Initialize random parameters
w = 1e-2 * mx.random.normal((num_features,))

# Define the loss function
def loss_fn(w):
    return 0.5 * mx.mean((X @ w - y)**2)

# Compute the gradient of the loss function
grad_fn = mx.grad(loss_fn)

# Training loop
tic = time.time()
for _ in range(num_iters):
    grad = grad_fn(w)
    mx.step(w, lr, grad)  # Update parameters using the gradient and learning rate
toc = time.time()

# Calculate final loss and error
loss = loss_fn(w)
error_norm = mx.sum((w - w_star)**2).item() ** 0.5

# Calculate throughput
throughput = num_iters / (toc - tic)

# Print results
print(
    f"Loss {loss:.5f}, L2 distance: |w-w*| = {error_norm:.5f}, "
    f"Throughput {throughput:.5f} (it/s)"
)

