# ============================================
# DAY 1: Building a single "neuron" from scratch
# Goal: teach a computer to find the line y = wx + b
# that best fits some data — using NO libraries
# like PyTorch/TensorFlow, just plain math (NumPy)
# ============================================

import numpy as np  # numpy lets us do fast math on lists of numbers (called "arrays")

# This makes our "random" numbers the same every time we run the code.
# Without this, results would be different each run — this makes it repeatable
# so you can compare your output to mine.
np.random.seed(42)

# ------------------------------------------------
# STEP 1: Create fake data to practice on
# ------------------------------------------------

# X = our input values. np.linspace(0, 10, 50) creates 50 evenly spaced
# numbers between 0 and 10 (like 0, 0.2, 0.4, ... up to 10)
X = np.linspace(0, 10, 50)

# y = our "correct answers". We SECRETLY decide the real formula is y = 2x + 1.
# We add some random noise (np.random.randn(50) * 1.5) so it's not a perfectly
# straight line — just like real-world data never is perfectly clean.
# IMPORTANT: the model will NOT be told w=2, b=1. It has to discover this itself.
y = 2 * X + 1 + np.random.randn(50) * 1.5

# ------------------------------------------------
# STEP 2: Start with a random, wrong guess
# ------------------------------------------------

# w = "weight" = the slope of our line (like "m" in y = mx + b from school)
# b = "bias" = where the line crosses the y-axis
# Both start as random numbers — the model has no idea what the right answer is yet.
w = np.random.randn()
b = np.random.randn()

# lr = "learning rate" = how big a step we take each time we correct our guess.
# Too big -> we overshoot and never settle. Too small -> learns very slowly.
lr = 0.01

# epochs = how many times we repeat the "guess -> check -> improve" loop.
epochs = 200

# ------------------------------------------------
# STEP 3: The training loop — this is the heart of machine learning
# ------------------------------------------------

for epoch in range(epochs):  # repeat this block 200 times

    # ---- FORWARD PASS ----
    # Using our CURRENT (possibly still wrong) w and b, predict y for every X
    y_pred = w * X + b

    # ---- LOSS (how wrong are we?) ----
    # Compare our prediction (y_pred) to the real answer (y).
    # We square the difference so negative and positive errors don't cancel out,
    # and so being "very wrong" counts as MUCH worse than being "a little wrong".
    # Then we average it across all 50 points to get ONE single "badness score".
    loss = np.mean((y_pred - y) ** 2)

    # ---- GRADIENTS (which direction should we adjust w and b?) ----
    # dw tells us: if we increase w slightly, does the loss go up or down?
    # db tells us the same thing but for b.
    # (This comes from calculus — don't worry about deriving it by hand yet,
    # just trust that this formula correctly points us toward less error)
    dw = np.mean(2 * (y_pred - y) * X)
    db = np.mean(2 * (y_pred - y))

    # ---- UPDATE (the actual "learning" step) ----
    # Nudge w and b a small amount (controlled by lr) in the direction
    # that REDUCES the loss. This is called "gradient descent".
    w -= lr * dw
    b -= lr * db

    # ---- PROGRESS CHECK ----
    # Every 20 loops, print out what's happening so we can watch it improve.
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

# ------------------------------------------------
# STEP 4: Final result
# ------------------------------------------------
# After 200 rounds of guessing and correcting, w and b should now be
# VERY close to the real secret values: w=2, b=1
print(f"\nFinal: w={w:.4f}, b={b:.4f} (target was w=2, b=1)")