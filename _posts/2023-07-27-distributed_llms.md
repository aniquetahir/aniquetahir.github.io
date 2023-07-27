---
layout: distill
title: training large language models using distributed resources
description: With the increasing popularity of large language models, it becomes increasingly important to use all available resources to gain an edge in training.
date: 2023-07-27
giscus_comments: true
related_posts: true

authors:
  - name: Anique Tahir
    url: "https://anique.org"
    affiliations:
      name: Arizona State University, Tempe, AZ

---

## Why do we need Distributed Training?
Large language models such as OpenAI's ChatGPT and Meta's Llama have gained popularity due to their usefulness. One of the major criticism of ChatGPT is that it's training parameters are ironically closed source. However, Meta's Llama models provide an Open Source alternative albeit being behind in quality. The NLP community has leveraged Llama to create variations which are intended to improve on the original model. Some improvements include uncensoring the model so that it is capable of responding in an unmoderated manner or finetuning it to provide better responses to instruct prompts. The possibilities of using open source LLMs are limitless. One simple application is to fine-tune Llama on a specific dataset. However, due to the size of the model, some of the larger versions may not fit on a single GPU. Hence, it is vital to learn how to train these models over a number of GPUs (multi-gpu) or a number of nodes each consisting of several GPUs (multi-node). 

## Basics of Distributed Training
In this post, I will focus on distributed training over a single node containing multiple GPUs. One does not need to train models on a GPU. In fact, if one were interested in using the cloud, TPUs are a promising alternative. However, TPUs are not sold to the public and I am a promoted of doing things on hardware which one has physical access to. Thus, I will focus on training on NVidia GPUs (the other alternative, AMD, does not have good support for machine learning as per current date). This post will be hands-on with code examples you can run. 

The first step is to check whether your rig has multiple GPUs and how they are linked together. The following command will return the status of the GPUs including the memory consumption and the processing power being consumed. It is handy to run this utility during training to get an idea of the resources being consumed (or not being consumed). In case the processing power is not 100%, it is indicative of a bottleneck in the training script which needs to be fixed.
```bash
nvidia-smi
```

Next, the following command helps to see how the GPUs are linked together:
```
nvidia-smi topo -m
```
Output:
```
	GPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	PHB	0-11		N/A		N/A
GPU1	PHB	 X 	0-11		N/A		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

NVidia cards allow being linked together using NVlink cables. This helps cross-GPU communication. However, as visible from the output, the cards need not be linked in such a way. Even if the GPUs are connected to seperate PCI ports on the same node. They can still communicate. However, cross-GPU communication will be slower. When designing a model for distributed training, the topology of the resources needs to be taken into consideration.

There are multiple ways to parallelize a model e.g. Data Parallelism, Pipeline Parallelism, etc. There are pros and cons of each strategy and they can be mixed together. 




 

## Limitations of current implementations


***
To be continued
(This post is a Work in Progress)

