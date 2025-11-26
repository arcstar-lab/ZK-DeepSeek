---
license: mit
base_model:
- deepseek-ai/DeepSeek-V3-0324
pipeline_tag: text-generation
tags:
- LLM
- zkSNARK
- ZKP
- SNARK
- Web3
- VerifiableAI
- int64
library_name: transformers
---

# 📘 **ZK-DeepSeek Model Card**

*A Verifiable Large Language Model with Zero-Knowledge Proofs*

For more details, please refer to our paper:  
📄 **Zero-Knowledge Proof Based Verifiable Inference of Models**  
https://arxiv.org/pdf/2511.19902

---

## Overview

**ZK-DeepSeek** is a *verifiable large language model (Verifiable LLM)* whose inference can be cryptographically proven using **zero-knowledge proofs (zkSNARKs)**.

For every inference step, the model produces a succinct mathematical proof that verifies:

* The output is computed faithfully by the intended model, and
* No model parameters or internal states are revealed.

This enables *trustless, provable, and privacy-preserving* AI inference suitable for high-stakes environments such as blockchain, Web3, finance, governance, and distributed systems.

This repository provides:

* A fully arithmeticized version of the model (Int64-based)
* Layer-wise model loading and computing
* Per-component zero-knowledge circuits (GeMM, RMSNorm, RoPE, Softmax, SiLU, MLA, MoE)
* A recursive SNARK pipeline

---

## Key Features

### **Verifiable Inference**

Every model output is accompanied by a zkSNARK proof guaranteeing correctness.

### **Zero-Knowledge Privacy**

The prover demonstrates correct computation without exposing any model parameters.

### **Full Arithmeticization**

All DeepSeek-like Transformer operations are converted into constraint-friendly circuits:

* GEMM (matrix multiplication)
* RMSNorm
* RoPE
* Softmax
* Sigmoid / SiLU
* MLA (Multi-Head Latent Attention)
* MoE (Mixture-of-Experts routing)

### **Recursive Proof Composition**

Tens of thousands of component proofs are folded into a single constant-sized proof.

---

## Model Details

| Item | Description |
|------|-------------|
| Architecture | DeepSeek-V3 style Transformer with MoE + MLA |
| Parameters | ~**671B** (quantized) |
| Quantization | Int64 / Int32 — exact arithmetic for zk circuits |
| Disk Size | ~**2.5 TB** (expanded integer representation) |
| Intended Use | Research & verifiable inference |

## Environment Requirements

### **Hardware**

Recommended:

* NVIDIA RTX 4090 / 5090 (preferred) * 1
* 64 GB RAM
* 6 TB NVMe SSD (800 G for virtual memory)

### **Software**

* Node.js 24.8 + TypeScript
* CUDA 12.9+
* Python 3.10
* o1js 2.10

---

## Quick Start

### **Generate Zero-Knowledge Witnesses**

Clone the repo:
```
git clone https://huggingface.co/arcstar-lab/ZK-DeepSeek
cd ZK-DeepSeek/inference
```

Install dependencies:
```
pip install -r requirements.txt
```

Compile CUDA kernels (-arch depends on your GPU):
```
nvcc -O3 -Xcompiler -fPIC -shared -o libint64gemm.so int64_gemm.cu -arch=sm_120
```

Download DeepSeek-V3 model weights and place them in:
```
/path/to/DeepSeek-V3
```

Convert weights to integer representation:
```
python3 ./convert2.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/ZK-DeepSeek-Demo1 --n-experts 256 --model-parallel 1
```

This also stores embedding data in the `zkdata` folder.

Run chat interface:
```
./runLLM.sh
<Input your message>
```

The inference log will be available in the `logs` folder, and intermediate states in `zkdata`.

![Inference result](https://raw.githubusercontent.com/arcstar-lab/ZK-DeepSeek/refs/heads/main/figures/result.png)

### **Generate Zero-Knowledge Proofs**

Move to `zk` folder and install ZK dependencies:
```
cd ../zk
npm install
npm run build
```

Start proof generation for vocabulary embeddings.
```
python3 runZK.py
```

To enable additional components, uncomment lines in `runZK.py` such as:
```
    # await taskExpertSelector_gate(0, 4)
    await taskEmbed()
    # await taskAttnNorm('attn_norm', 0, 0)
    # await taskAttnNorm('q_norm')
    # await taskRope_pe()
    # await taskSoftmax('scores')
    # await taskSigmoid('gate')
    # await taskExpertSelector('gate', 0, 3)
    # await taskGemm('wkv_a1', 0, 0, 7168, 512, 112)
```

---

## Safety & Limitations

### **Security Guarantees**

* Proofs ensure inference correctness
* Model parameters remain private
* Inference cannot be forged or shortcut
* Compatible with on-chain trustless environments

### **Limitations**

* Proving is computationally heavy but can be accelerated by GPU

---

## License

MIT

---

## Contact

**Author:** Edward Wang
**Email:** [wyx96922@gmail.com](mailto:wyx96922@gmail.com)

**Our next priority is to develop a GPU-accelerated version of ZK-DeepSeek, which we anticipate will yield performance improvements by several orders of magnitude. We are actively seeking funding and collaborators to help advance this line of research. If you are interested in supporting or partnering with us, we warmly invite you to get in touch by email.**