# **Mathomong AI Club**

![mathomong](mathomong.png)

# Calculus in Neural Networks: A Beginner's Guide

---

## 📘 What You Will Learn

By the end of this guide, you’ll understand:

* Why neural networks need calculus
* What derivatives really mean (intuitively!)
* How gradients guide learning
* What backpropagation is and why it's powerful
* How everything fits together in training a neural network

---

# 🌟 Part 1: Why Calculus?

Neural networks have **parameters** (weights and biases).
Training means:

> **Finding the best values for these parameters.**

To do that, we ask:

> “If I change this weight slightly, how will the error change?”

This is exactly what **calculus** answers.

### ⚡ Calculus = The Mathematics of Change

Neural networks improve by *slowly adjusting parameters in the direction that reduces error*.

To know the best direction → we need **derivatives**.
To know how big each adjustment should be → we use **gradients**.

---

# 🟦 Part 2: Derivatives — The Core Idea

### 🔑 What is a derivative (intuition)?

A derivative answers:

> **“If I change input a tiny bit, how does the output change?”**

Example:

```
y = x²
dy/dx = 2x
```

Which means:

* If x = 3, then a tiny change in x causes y to change by about **6 times** that amount.

### Why neural networks need this

A neural network wants to know:

> “If I increase/decrease weight W by a tiny amount, does loss go up or down?”

That’s a derivative.

---

# 🧩 Part 3: Loss Functions (Where Calculus Happens)

Neural networks don’t know what correct weights are.
So we create a **loss function**, which measures *how wrong the network is*.

Example (mean squared error):

```
Loss = (prediction − target)²
```

The goal is:

> Minimize loss by adjusting weights.

To do that, we need:

```
d(Loss)/d(weight)
```

---

# 📉 Part 4: Gradient Descent — Learning Through Calculus

Gradient Descent is the algorithm that uses calculus to update weights:

```
new_weight = old_weight − learning_rate × gradient
```

### What is a gradient?

A gradient is:

* A **vector of all partial derivatives**
* It tells us the **direction of steepest increase**
* So we move **in the opposite direction** to minimize loss

Think of it like sliding down a hill to reach the lowest valley.

---

# 🎯 Part 5: Partial Derivatives — Essential for Multi-Input Systems

A neuron receives many inputs:

```
z = W1x1 + W2x2 + … + Wnxn + b
```

The loss depends on *each weight separately*, so we compute:

```
∂Loss/∂W1
∂Loss/∂W2
...
∂Loss/∂Wn
```

These are partial derivatives — calculus for functions with many variables.

---

# 🔄 Part 6: The Chain Rule — The Engine Behind Backpropagation

### Why the chain rule?

In a neural network:

```
input → layer1 → layer2 → layer3 → loss
```

Each layer’s output depends on the previous layer’s output.

To compute derivative at early layers, you must "chain" effects backward.

Chain Rule says:

> If A affects B, and B affects C, then
> A affects C by multiplying their derivatives.

Formally:

```
dC/dA = dC/dB × dB/dA
```

In a neural network:

```
dLoss/dW = dLoss/dOutput × dOutput/dZ × dZ/dW
```

That's backpropagation.

---

# 🔥 Part 7: Backpropagation — The Brain of Training

Backpropagation applies the chain rule across layers to compute gradients efficiently.

### Steps:

1. **Forward Pass**

   * Compute predictions
   * Compute loss

2. **Backward Pass**

   * Compute gradients layer-by-layer using chain rule

3. **Update Weights**

   * Using gradient descent

Backpropagation gives exact derivatives for millions of parameters using simple repeated calculus rules.

---

# 🧠 Part 8: Activation Functions and Calculus

Activation functions introduce **nonlinearity**, but they must be differentiable for training.

Common ones:

### 1. **ReLU**

```
ReLU(x) = max(0, x)
d/dx = 0 (x<0), 1 (x>0)
```

### 2. **Sigmoid**

```
sigmoid(x) = 1 / (1+e^-x)
derivative = sigmoid(x)(1−sigmoid(x))
```

### 3. **Tanh**

```
derivative = 1 − tanh²(x)
```

The fact that all these have known derivatives makes training possible.

---

# 🧮 Part 9: Example — Calculus in One Neuron

Consider a simple neuron:

```
z = Wx + b
prediction = sigmoid(z)
loss = (prediction − target)²
```

To train it:

We compute:

1. ∂Loss/∂prediction
2. ∂prediction/∂z
3. ∂z/∂W and ∂z/∂b

Using chain rule:

```
∂Loss/∂W = ∂Loss/∂prediction × ∂prediction/∂z × ∂z/∂W
```

This is the logic for all neural networks — just repeated many times.

---

# 🌊 Part 10: Calculus Shapes the Learning Landscape

Think of the loss function as a mountain valley.
Training is the process of:

* Standing somewhere on the mountain (initial weights)
* Calculating the slope beneath your feet (gradient)
* Taking a step downhill
* Repeating until reaching a low point

**Calculus = Understanding the slope**
**Optimizers = Deciding how to move**
**Networks = Learning the landscape**

---

# 💡 Key Takeaways

1. **Neural networks learn by minimizing error**
2. **Calculus tells us how changing weights affects error**
3. **Derivatives measure sensitivity (change in change)**
4. **Gradients show direction for learning**
5. **Chain rule lets information flow backward**
6. **Backpropagation = efficient gradient computation**
7. **Learning = adjusting millions of parameters using calculus**

---

# 🚀 What’s Next?

To build deeper understanding:

* Watch 3Blue1Brown’s *“Backpropagation Explained”*
* Practice computing simple derivatives by hand
* Build a neural network from scratch in Python/Numpy
* Visualize gradient descent on simple 2D functions
* Join the **Mathomong AI Club** to grow with others!

---

*"In learning, you will teach, and in teaching, you will learn."* — Phil Collins

**Welcome to the Mathomong AI Club!**
