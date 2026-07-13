
---

# DLWP Chapter 3 — Tensor Fundamentals (PyTorch)

## 1. Tensor Memory Model

### Core Idea

A tensor consists of **two parts**:

```text
Tensor
│
├── Metadata
│     ├── Shape
│     ├── Stride
│     ├── Dtype
│     ├── Device
│     └── Storage Offset
│
└── Storage (Actual Memory)
```

Metadata tells PyTorch **how to interpret memory**.

Storage contains the actual values.

---

## Shape

**Definition**

Number of elements along each dimension.

Example

```python
x = torch.randn(3,4)
```

Shape

```python
(3,4)
```

---

## Storage

Underlying contiguous memory buffer.

Multiple tensors may share the same storage.

---

## Stride ⭐

### Definition

Number of memory locations to skip to move one step in a dimension.

Example

```python
x = torch.arange(12).view(3,4)
```

Shape

```
(3,4)
```

Stride

```
(4,1)
```

Meaning

Move

* Down one row → skip 4 values
* Right one column → skip 1 value

---

## Storage Offset

Starting position inside storage.

Useful when slicing.

---

## Contiguous Memory

A tensor is contiguous if its elements are stored in memory exactly as its shape expects.

Many operations require contiguous memory.

---

# view()

### Purpose

Changes tensor shape **without copying memory**.

Requirements

✔ Same number of elements

✔ Tensor must be contiguous

---

Example

```python
x.view(2,6)
```

---

# reshape()

Similar to view()

Difference

```text
view()
↓

Fails if non-contiguous

--------------------

reshape()

↓

Creates copy if necessary
```

---

# transpose()

Swaps dimensions.

Does NOT move memory.

Changes

* Shape
* Stride

NOT storage.

---

# permute()

Generalization of transpose.

Reorders dimensions arbitrarily.

Example

```
N,C,H,W

↓

N,H,W,C
```

Again

Storage stays same.

Metadata changes.

---

# contiguous()

Creates new contiguous copy.

Required before

```
view()
```

on non-contiguous tensors.

---

# Broadcasting

Allows arithmetic on tensors with different shapes.

Rules

Compare dimensions from right.

Each dimension must satisfy

* Equal
* One is 1

Otherwise

Broadcast fails.

---

# Unsqueeze

Adds dimension.

```
(5,)

↓

(1,5)
```

---

# Squeeze

Removes dimensions of size 1.

```
(1,5,1)

↓

(5)
```

---

# Advanced Indexing

Learned

* Boolean indexing
* Fancy indexing
* Masking

---

# Active Recall

Without looking:

1. Difference between shape and storage?

2. Why does view() fail?

3. Difference between reshape() and view()?

4. Does permute move memory?

5. What changes after transpose?

6. What is stride?

7. Why does contiguous() exist?

8. What does storage_offset represent?

9. Broadcasting rules?

10. When is squeeze dangerous?

---

# Interview Questions

### Beginner

What is a tensor?

Difference between tensor and ndarray?

---

### Intermediate

Difference between

view

reshape

flatten

---

Why does

```python
x.permute(...).view(...)
```

sometimes fail?

---

Difference between

transpose

permute

---

Explain broadcasting.

---

### Advanced

What is stride?

How does PyTorch access tensor elements?

Why does permute not copy memory?

How does contiguous() work internally?

---

# Common Mistakes

❌ Thinking permute copies memory

❌ Confusing reshape with view

❌ Ignoring stride

❌ Forgetting contiguous()

---

# Practical Applications

CNNs

Image preprocessing

Transformer attention

Batch creation

---

---

# DLWP Chapter 4 — Real World Data Representation

---

# Goal

Convert real-world data into tensors.

Everything in deep learning follows

```
Raw Data

↓

Numerical Representation

↓

Tensor

↓

Neural Network
```

---

# Image Representation

RGB

```
Height

×

Width

×

Channels
```

Usually

```
(C,H,W)
```

in PyTorch.

---

# Tabular Data

Rows

↓

Samples

Columns

↓

Features

```
(N,F)
```

---

# Time Series

Sequential observations.

Representation

```
(Time, Features)
```

Later converted into

```
(Window, Features)
```

for DL models.

---

# Text Representation

Pipeline

```
Sentence

↓

Tokenizer

↓

Vocabulary

↓

Integer IDs

↓

One-Hot Encoding

↓

Embeddings
```

---

# Character Encoding

Vocabulary

```
a-z

digits

symbols
```

Small vocabulary.

Long sequences.

---

# Word Encoding

Vocabulary

Entire words.

Large vocabulary.

Shorter sequences.

Problem

Huge one-hot vectors.

Unknown words.

---

# One-Hot Encoding

Example

```
Dog

↓

[0 0 1 0 0]
```

Properties

Sparse

High dimensional

No semantic similarity.

---

# Vocabulary

Dictionary

```
word

↓

index
```

Example

```
cat → 0

dog → 1

tree → 2
```

---

# Embeddings ⭐

Core Idea

Instead of

```
Dog

↓

50000 dimensions
```

learn

```
Dog

↓

128 dense numbers
```

Advantages

Smaller

Semantic meaning

Learnable

---

# Why Embeddings?

One-hot

```
King

Queen

Dog
```

All equally different.

Embeddings

```
King ≈ Queen

Dog far away
```

Semantic information emerges.

---

# Subword Tokenization

Motivation

Word vocabulary becomes enormous.

Solution

Split words.

Example

```
playing

↓

play

ing
```

Popular methods

* BPE
* WordPiece
* SentencePiece

Modern LLMs use subword tokenization.

---

# Active Recall

1. Why do neural networks need tensors?

2. Character vs word encoding?

3. Advantages of embeddings?

4. Why is one-hot inefficient?

5. What problem does BPE solve?

6. Why not use words only?

7. Why not use characters only?

8. Pipeline from sentence to tensor?

9. What is vocabulary?

10. Why are embeddings learnable?

---

# Interview Questions

### Beginner

What is one-hot encoding?

Difference between word and character encoding?

---

### Intermediate

Why are embeddings superior?

How are vocabularies built?

Why can't we directly feed text into a neural network?

---

### Advanced

Explain BPE.

Difference between

One-Hot

Word2Vec

GloVe

FastText

Transformer Embeddings

Why do LLMs use tokenizers instead of words?

---

# Common Mistakes

❌ Thinking one-hot contains meaning

❌ Thinking embeddings are predefined

❌ Confusing tokenizer with embedding layer

❌ Treating tokenization and encoding as the same step

---

# Mental Models

### Tensor

```
Tensor

=

Storage

+

Metadata
```

---

### NLP Pipeline

```
Raw Text
      │
      ▼
Tokenizer
      │
      ▼
Vocabulary
      │
      ▼
Token IDs
      │
      ├────────► One-Hot (historical)
      │
      └────────► Embedding Layer (modern)
```

---

# 🔥 20 Lightning Revision Questions (2-Minute Self-Test)

1. Why does `view()` require contiguous memory?
2. What changes after `permute()`?
3. What is a stride of `(4,1)` telling you?
4. `view()` vs `reshape()`?
5. Why does broadcasting work?
6. What is `storage_offset`?
7. Does `transpose()` copy data?
8. Why call `.contiguous()`?
9. How is tabular data represented as a tensor?
10. How is time-series data represented?
11. Why is one-hot encoding sparse?
12. Why does vocabulary size matter?
13. Character vs word tokenization: trade-offs?
14. Why are embeddings more powerful than one-hot vectors?
15. What semantic information can embeddings capture?
16. What problem do unknown (`<UNK>`) words create?
17. Why did subword tokenization become the standard?
18. What is the difference between a **tokenizer** and an **embedding layer**?
19. Why can't raw text be fed directly into a neural network?
20. Explain the full pipeline from raw text to transformer input in under one minute.

---
