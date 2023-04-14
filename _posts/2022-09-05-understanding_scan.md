---
layout: distill
title: understanding jax's scan function
description: jax.lax.scan is a function which allows jit-able loops
date: 2022-09-05
giscus_comments: true
related_posts: true

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

According to the jax [documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html), the following code is 
essentially a translation of the function in pythonic form:

```python3
def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)
```

The code may be a little convoluted to understand. A simpler way to understand it is to look at some simple examples. The `scan` 
function takes three parameters and scans over the third argument. The first arguments is a function to execute over each scan iteration.
The second argument is some `pytree` structure which we initially start from. Lets look at a simple example.

```python3
scan(lambda x, y: (x+y, y+2,), 0, [1, 2, 3])
```

The above code essentially scans through the list `[1, 2, 3]` and for each element, returns a tuple consisting of the previous element and the current element. The output of the above code will be:
```python
(6, [3, 4, 5])
```
Here `x` is the carry argument which is initialized to `0`. `y` is each element of the array `[1, 2, 3]` passed sequentially to the function. In the first pass `x` is `0` and `y` is 1. The return value is `(x+y, y+2,) = (0+1,  1+2,) = (1, 3,)`. For the next iteration, `x` is `1`, since `1` is the value of the carry returned in the last iteration and `y` is `2`, since `2` is the next value in the input array. Thus, `(x+y, y+2,) = (1+2,  2+2,) = (3, 4,)`. In the next iteration, the final value of the carry is `6`, while the final `y+2` is `5`. Thus, the `scan` function returns `6`(the carry) and `[3, 4, 5]`(all the `y+2` concatenated)

Here we used a simple example, but the scan function can be used over more complicated arguments, such as the training loop of a neural network. This makes it possible to jit compile the entire training process, resulting in a large gain in training speed.




