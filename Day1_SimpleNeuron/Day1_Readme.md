# Day 1 — Building a Single Neuron from Scratch

> *"Before building deep neural networks, you need to understand the smallest unit they are made of: a single neuron."*

## 🎯 Learning Objectives

By the end of this lesson, you will understand:

* What a neuron is
* How a neuron makes predictions
* What **weights** and **biases** are
* How to measure prediction error using **Mean Squared Error (MSE)**
* How **Gradient Descent** helps a model learn from its mistakes
* The complete machine learning training loop

---

# Why Start Here?

Every modern neural network—whether it's ChatGPT, a self-driving car, or an image classifier—is built from simple mathematical operations.

At its core, a neuron performs one task:

> **Take an input, make a prediction, compare it with the correct answer, and improve itself.**

Understanding this single process makes everything that follows in deep learning much easier.

---

# The Problem

Suppose we have data that follows a pattern like this:

```
x → y

1 → 3
2 → 5
3 → 7
4 → 9
...
```

There is a hidden relationship between the input (`x`) and output (`y`).

Our goal is to let the computer **discover** that relationship instead of hardcoding it.

---

# The Neuron

A single neuron is simply a mathematical equation:

```
y = wx + b
```

Where:

* **x** = Input
* **w** = Weight (controls the slope)
* **b** = Bias (shifts the line up or down)
* **y** = Prediction

Initially, the neuron has no idea what the correct values of `w` and `b` should be.

It starts with random guesses.

---

# Step 1 — Create Training Data

Instead of using a real dataset, we generate our own.

```python
X = np.linspace(0, 10, 50)
```

This creates 50 evenly spaced input values.

Then we secretly create the correct relationship:

```python
y = 2 * X + 1
```

To make the problem realistic, random noise is added.

Real-world datasets are rarely perfect.

---

# Step 2 — Initialize Random Parameters

The model begins with random values.

```python
w = np.random.randn()
b = np.random.randn()
```

At this point, the predictions are usually very inaccurate.

That is completely normal.

Learning starts from being wrong.

---

# Step 3 — Make Predictions

Using the current values of `w` and `b`, the neuron predicts every output.

```
Prediction = w × X + b
```

This process is called the **Forward Pass**.

The model is simply answering:

> "Based on what I currently know, what do I think the outputs are?"

---

# Step 4 — Measure the Error

Now compare the predictions with the correct answers.

We use **Mean Squared Error (MSE)**:

```
Loss = Average((Prediction − Actual)²)
```

Why square the error?

* Negative and positive errors don't cancel each other.
* Large mistakes are penalized more heavily.
* The function becomes smooth and easy to optimize.

A lower loss means the model is performing better.

---

# Step 5 — Compute the Gradients

The gradients answer two important questions:

* Should the weight increase or decrease?
* Should the bias increase or decrease?

They are calculated using:

```
dw
db
```

Think of the gradient as a compass.

It tells the model which direction reduces the error.

---

# Step 6 — Learn with Gradient Descent

Once the gradients are known, we update the parameters.

```
w = w − learning_rate × dw

b = b − learning_rate × db
```

Notice the subtraction.

We move **against** the gradient because the gradient points toward increasing error.

Gradient Descent moves toward lower error.

---

# Step 7 — Repeat

Learning doesn't happen in one update.

The model repeats this cycle many times:

```
Predict
      ↓
Measure Error
      ↓
Compute Gradients
      ↓
Update Parameters
      ↓
Repeat
```

Each repetition is called an **epoch**.

Over time, the loss decreases and the predictions improve.

---

# Training Loop

The complete learning process can be summarized as:

```
Initialize weights randomly
        ↓
Forward Pass
        ↓
Compute Loss
        ↓
Compute Gradients
        ↓
Update Parameters
        ↓
Repeat
```

This loop is the foundation of almost every machine learning algorithm.

---

# Expected Output

As training progresses, you should notice:

* Loss continuously decreasing
* Weight moving closer to the correct value
* Bias improving over time
* Predictions becoming increasingly accurate

Example:

```
Epoch 0
Loss = 112.61

↓

Epoch 180
Loss = 1.91
```

A decreasing loss indicates that the model is learning.

---

# Key Terms

| Term          | Meaning                                              |
| ------------- | ---------------------------------------------------- |
| Input (X)     | Data given to the model                              |
| Weight (w)    | Determines how strongly the input affects the output |
| Bias (b)      | Shifts the prediction upward or downward             |
| Prediction    | Model's estimated output                             |
| Loss          | Measures how wrong the prediction is                 |
| Gradient      | Indicates how to adjust the parameters               |
| Learning Rate | Controls the size of each update                     |
| Epoch         | One complete pass through the training data          |

---

# What You Learned Today

By completing this lesson, you have implemented a learning algorithm from scratch.

Without using TensorFlow or PyTorch, you learned how a model:

* Makes predictions
* Measures its error
* Computes gradients
* Updates its parameters
* Improves over time

These five steps form the foundation of deep learning.

Everything from logistic regression to convolutional neural networks and transformers relies on the same fundamental learning process.

---

# Practice Exercises

Try the following experiments:

* Increase the number of epochs and observe the results.
* Change the learning rate to see how training speed changes.
* Remove the random noise from the dataset.
* Create a new hidden equation (for example, `y = 5x - 8`) and see if the model can learn it.
* Plot the data points and the learned line using Matplotlib.

---

# Summary

Today you built your very first neuron from scratch.

Although simple, this neuron demonstrates the complete machine learning workflow:

> **Predict → Measure Error → Compute Gradients → Update → Repeat**

Mastering this cycle is the first step toward understanding how modern neural networks learn.

In the next lesson, you'll extend this idea from **one input feature** to **multiple input features**, allowing a neuron to learn from more complex data.
