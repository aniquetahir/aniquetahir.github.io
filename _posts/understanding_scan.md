---
layout: distill
title: understanding Jax's scan function
description: jax.lax.scan is a function which allows jit-able loops
date: 2022-09-05

authors:
  - name: Anique Tahir
    url: "https://cat.ninja"
    affiliations:
      name: Arizona State University, Tempe, AZ

---

## Why use `jax.lax.scan`

Jax is a neural network library used mostly by Google. Jax converts all your implementation into a graph which is executed
on your CPU, GPU or TPU. There are two main advantages of using Jax for your implementation:
- It is comparatively faster than it's competitor Pytorch
- It allows for significantly easier implementation for low level neural network concepts (albeit being harder for high-level
- ideas)

Jax allows you to `jit` your functions. `jit` stands for Just-In-Time compilation. This makes your function significantly fast
since it is compiled into something native to the GPU<d-footnote>or any device you compile for.</d-footnote>. However, the drawback 
is that the amount of memory that your function will use, has to be pre-specified. This means that functions containing loops
have to be changed, since the length of the loop can be arbitrary. `jax.lax.scan` allows you to get around this limitation by
allowing you to define a loop with pre-specified length. But how does it work? 

***

