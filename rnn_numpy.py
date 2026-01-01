import numpy as np

np.random.seed(0)

input_size = 1
hidden_size = 8
output_size = 1
learning_rate = 0.01

Wxh = 0.01 * np.random.randn(hidden_size, input_size)
Whh = 0.01 * np.random.randn(hidden_size, hidden_size)
Why = 0.01 * np.random.randn(output_size, hidden_size)
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))

def tanh(x):
    return np.tanh(x)

def tanh_deriv(h):
    return 1 - h ** 2

def forward(inputs):
    hs = {}
    hs[-1] = np.zeros((hidden_size, 1))
    ys = []
    for t in range(len(inputs)):
        x = inputs[t].reshape(input_size, 1)
        hs[t] = tanh(Wxh @ x + Whh @ hs[t-1] + bh)
        y = Why @ hs[t] + by
        ys.append(y)
    return ys, hs

def compute_loss(ys, targets):
    loss = 0.0
    for t in range(len(targets)):
        diff = ys[t] - targets[t].reshape(output_size, 1)
        loss += float((diff ** 2)[0, 0])
    return loss

def backward(inputs, targets, ys, hs):
    global Wxh, Whh, Why, bh, by
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)
    dh_next = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = ys[t] - targets[t].reshape(output_size, 1)
        dWhy += dy @ hs[t].T
        dby += dy

        dh = Why.T @ dy + dh_next
        dh_raw = dh * tanh_deriv(hs[t])
        dbh += dh_raw
        dWxh += dh_raw @ inputs[t].reshape(1, input_size)
        dWhh += dh_raw @ hs[t-1].T
        dh_next = Whh.T @ dh_raw

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy
    bh -= learning_rate * dbh
    by -= learning_rate * dby

# Training data
inputs = [np.array([i]) for i in range(3)]
targets = [np.array([i+1]) for i in range(3)]

# Training loop
for epoch in range(500):
    ys, hs = forward(inputs)
    loss = compute_loss(ys, targets)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')
    backward(inputs, targets, ys, hs)

# Final evaluation
ys, hs = forward(inputs)
print("\nFinal Predictions:")
for t in range(len(inputs)):
    print(f"Input: {inputs[t][0]}, Target: {targets[t][0]}, Predicted: {ys[t][0,0]:.4f}")
