
---

# 1. Tensor

## Definition

A multidimensional array used to represent data in PyTorch.

```text
Tensor

=

Storage

+

Metadata
```

Metadata

* Shape
* Stride
* Dtype
* Device
* Storage Offset

Storage

* Actual memory.

---

## Tensor Creation

```python
torch.tensor()
```

Recommended constructor.

Other creators

```python
zeros()
ones()
empty()
rand()
randn()
arange()
linspace()
```

Know what each returns.

---

# 2. Tensor Operations

## Element-wise

```python
*
+
-
/
```

Operate element by element.

---

## Matrix Multiplication

```python
torch.matmul()
@
```

Uses linear algebra rules.

---

## Shape Operations

| Function  | Purpose                       |
| --------- | ----------------------------- |
| view      | New view of contiguous memory |
| reshape   | View if possible, else copy   |
| flatten   | Collapse dimensions           |
| transpose | Swap two dimensions           |
| permute   | Arbitrary dimension reorder   |
| unsqueeze | Add dimension of size 1       |
| squeeze   | Remove dimensions of size 1   |

---

# 3. Broadcasting

Rules

Compare dimensions from right.

Compatible if

* equal
* one dimension is 1

Otherwise

Broadcast fails.

---

# 4. Computational Graph ⭐

Forward pass builds

```text
Tensor

↓

Operation

↓

Operation

↓

Loss
```

PyTorch records every operation.

Nothing is differentiated yet.

---

# 5. Leaf Tensor

Created directly by user.

```python
x = torch.tensor(..., requires_grad=True)
```

Leaf tensors store gradients.

---

# 6. requires_grad

```python
requires_grad=True
```

Meaning

> Track operations involving this tensor by building a computation graph.

Use for trainable parameters.

---

# 7. backward()

Runs backpropagation.

Traverses graph backwards.

Computes

```text
∂Loss
─────
 ∂Parameter
```

Stores result in

```python
.grad
```

---

# 8. grad

Stores computed gradient.

Example

```python
w.grad
```

NOT

weight value.

It stores

```text
∂Loss
─────
 ∂w
```

---

# 9. grad_fn

Every non-leaf tensor remembers

> Which operation created me?

Example

```python
MulBackward

AddBackward

ReluBackward
```

Autograd uses these nodes to traverse the graph.

---

# 10. zero_grad()

PyTorch accumulates gradients.

Therefore

```python
optimizer.zero_grad()
```

clears previous gradients before the next backward pass.

---

# 11. detach()

Cuts tensor from computation graph.

Same data.

No gradients.

Used for

* Logging
* NumPy conversion
* Visualization

---

# 12. torch.no_grad()

Temporarily disables graph construction.

Used during

* Inference
* Validation
* Testing

Advantages

* Faster
* Lower memory

---

# 13. nn.Module

Base class for all neural networks.

Provides

* Parameters
* forward()
* GPU movement
* Saving/loading
* train()/eval()

Examples

```python
Linear

Conv2d

LSTM

Sequential
```

---

# 14. forward()

Never call

```python
model.forward(x)
```

Always

```python
model(x)
```

because

```text
model()

↓

__call__()

↓

forward()
```

---

# 15. Dataset

Responsible for

```text
One sample
```

Implements

```python
__len__()

__getitem__()
```

---

# 16. DataLoader

Responsible for

```text
Batching

Shuffling

Parallel loading
```

---

# 17. Optimizer

General update

```text
Parameter

↓

Parameter − lr × Gradient
```

Examples

* SGD
* Adam

---

# 18. Standard Training Loop ⭐⭐⭐⭐⭐

```python
optimizer.zero_grad()

pred = model(X)

loss = criterion(pred,y)

loss.backward()

optimizer.step()
```

Know WHY every line exists.

---

# Complete Training Pipeline

```text
Raw Data
      │
      ▼
Dataset
      │
      ▼
DataLoader
      │
      ▼
Model
      │
      ▼
Prediction
      │
      ▼
Loss
      │
      ▼
backward()
      │
      ▼
.grad
      │
      ▼
optimizer.step()
```

---

# Interview Questions

## Easy

* What is a tensor?
* Difference between tensor and NumPy array?
* Difference between Dataset and DataLoader?
* What is a batch?
* Epoch vs Iteration?

---

## Medium

* Why does PyTorch use computation graphs?
* What is a leaf tensor?
* Difference between `.grad` and `.grad_fn`?
* Difference between `detach()` and `torch.no_grad()`?
* Why call `zero_grad()`?
* Why call `model(x)` instead of `forward()`?

---

## Advanced

* Explain Autograd internally.
* How does backpropagation work?
* Why are gradients stored only for leaf tensors?
* Why does `reshape()` sometimes copy memory?
* How does broadcasting work?
* Why does `view()` require contiguous memory?

---

# 2-Minute Revision

Can you answer these without notes?

1. What is a computation graph?
2. What is a leaf tensor?
3. Difference between `.grad` and `.grad_fn`?
4. Why is `zero_grad()` required?
5. Difference between `detach()` and `torch.no_grad()`?
6. Difference between `view()` and `reshape()`?
7. Dataset vs DataLoader?
8. Why inherit from `nn.Module`?
9. Explain the training loop line by line.
10. What exactly does `backward()` compute?

---
