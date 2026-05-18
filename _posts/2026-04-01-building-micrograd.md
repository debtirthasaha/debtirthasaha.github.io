---
layout: post
title: "micrograd: a scalar-valued autograd engine"
date: 2026-04-01 10:00:00
description: A 150-line autograd engine that supports +, *, **, tanh, exp, and a tiny MLP on top.
tags: autograd backprop
categories: deep-learning
giscus_comments: false
chart:
  plotly: true
toc:
  sidebar: left
---

A scalar-valued autograd engine. Every value is a Python `float` wrapped in a `Value` object that knows what produced it. Operator overloads build a DAG implicitly. `backward()` topologically sorts the DAG and runs each node's local gradient rule in reverse. About 150 lines total.

## The `Value` class

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
```

- `data`: the forward scalar.
- `grad`: filled in by backward, starts at 0.
- `_prev`: parents in the DAG.
- `_op`: string label (debugging only).
- `_backward`: closure each operation sets; default no-op for leaves.

## Operator overloads register local gradient rules

Addition:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad  += 1.0 * out.grad
        other.grad += 1.0 * out.grad
    out._backward = _backward
    return out
```

Three things at once: forward arithmetic, DAG construction (`(self, other)` as parents), and the local rule (`d(a+b)/da = 1`, `d(a+b)/db = 1`).

Multiplication: same shape, different local rule.

```python
def _backward():
    self.grad  += other.data * out.grad
    other.grad += self.data  * out.grad
```

`tanh`:

```python
def tanh(self):
    t = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out
```

`**` and `exp` are the same pattern.

## Why `+=` and not `=`

If a node feeds into multiple downstream nodes, the chain rule sums contributions over all paths. `+=` is exactly that sum.

This is why `optimizer.zero_grad()` exists in PyTorch. Gradients accumulate by design; you have to clear them between training steps or you accumulate gradients across batches.

## Backward via topo-sort

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

Topo sort guarantees that when `v._backward()` runs, `v.grad` already has its final value — every downstream node has already pushed into it. Walking in reverse without the sort gives stale gradients.

The base case `self.grad = 1.0` is the seed: the gradient of the final scalar with respect to itself is 1.

## A 2-3-3-1 MLP, no PyTorch

A `Neuron` is a list of weight `Value`s, a bias `Value`, and a `tanh`. A `Layer` is a list of `Neuron`s. An `MLP` is a list of `Layer`s.

Training loop:

```python
for k in range(50):
    ypred = [n(x) for x in xs]
    loss  = sum((yp - y)**2 for yp, y in zip(ypred, ys))

    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    for p in n.parameters():
        p.data -= 0.05 * p.grad
```

`(yp - y)**2`, `sum`, every neuron's `tanh` — all of it constructs nodes in the same `Value` DAG. `loss.backward()` walks the whole thing.

41 parameters. The actual training trajectory on the four-point demo, every step:

```plotly
{"data":[{"x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49],"y":[3.764166,3.514946,3.152909,2.638927,2.02037,1.407935,0.914896,0.591433,0.401572,0.290094,0.22114,0.175845,0.144458,0.121721,0.104642,0.091425,0.080942,0.072454,0.065461,0.059612,0.054657,0.050411,0.046738,0.043532,0.040712,0.038214,0.035988,0.033992,0.032194,0.030566,0.029087,0.027736,0.026498,0.025361,0.024312,0.023343,0.022443,0.021607,0.020828,0.020101,0.01942,0.018782,0.018182,0.017618,0.017086,0.016584,0.016109,0.015659,0.015233,0.014828],"type":"scatter","mode":"lines+markers","line":{"color":"#EF553B","width":2},"marker":{"size":4},"hovertemplate":"step %{x}<br>loss %{y:.4f}<extra></extra>","name":"loss"}],"layout":{"title":{"text":"Training loss, 41-parameter MLP on 4 points"},"xaxis":{"title":"step"},"yaxis":{"title":"sum-of-squares loss","type":"log"},"height":420,"margin":{"l":70,"r":30,"t":60,"b":50},"showlegend":false}}
```

3.76 → 0.015 in 50 steps. Steep drop for the first ~10 steps as the network finds the rough direction, then a slow log-linear decline as it refines. Every gradient was computed by my own code.

## Why scalar-valued

PyTorch operates on tensors and broadcasts the same chain rule across millions of elements per op. The math is identical; tensor ops just batch it. Once the scalar engine works, the upgrade to a tensor engine is engineering, not concept.

Code: [github.com/debtirthasaha/micrograd-from-scratch](https://github.com/debtirthasaha/micrograd-from-scratch).
