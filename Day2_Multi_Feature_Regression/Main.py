import numpy as np

np.random.seed(42)

# Fake data: 100 houses, 3 features each (size, rooms, age)
n_samples = 100
n_features = 3

X = np.random.rand(n_samples, n_features) * 10  # random values 0-10 for each feature

# The REAL relationship we want the model to discover:
# price = 3*size + 5*rooms - 2*age + 10  (plus some noise)
true_weights = np.array([3, 5, -2])
true_bias = 10
y = np.dot(X, true_weights) + true_bias + np.random.randn(n_samples) * 2

# Initialize parameters randomly
w = np.random.randn(n_features)   # now an array of 3 weights, not just one number
b = np.random.randn()
lr = 0.01
epochs = 500

for epoch in range(epochs):
    # Forward pass
    y_pred = np.dot(X, w) + b

    # Loss: Mean Squared Error
    loss = np.mean((y_pred - y) ** 2)

    # Gradients
    dw = np.dot(X.T, (y_pred - y)) * (2 / n_samples)
    db = np.mean(2 * (y_pred - y))

    # Update
    w -= lr * dw
    b -= lr * db

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}, w={w}, b={b:.4f}")

print(f"\nFinal: w={w}, b={b:.4f}")
print(f"Target was: w={true_weights}, b={true_bias}")