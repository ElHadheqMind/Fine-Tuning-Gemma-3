# Fine-Tuning Gemma 3 270M (or Higher) on CPU/GPU with LoRA: A Complete Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DUnWNxLMxL3WprUqhyLayPxVr5KffTsC?usp=sharing)

![Fine-Tuning Hero Image](cpu%20super%20sin.png)

*Ready to Fine-Tune on CPU? Let's go!*

---

## Introduction

Fine-tuning large language models has traditionally required expensive GPU hardware, leaving many developers and researchers on the sidelines. But what if I told you that you could fine-tune Google's Gemma 3 model using just your laptop's CPU?

In this comprehensive guide, we'll walk through the complete process of fine-tuning **Gemma 3 270M** using **Low-Rank Adaptation (LoRA)** — a parameter-efficient technique that makes fine-tuning accessible to everyone. Whether you're a student learning ML, a domain expert adapting models for specialized tasks, or an ML engineer looking for practical implementation details, this guide has you covered.

---

## Who Is This Guide For?

- **Students & Researchers**: Learn the complete workflow from data creation to model deployment
- **ML Engineers**: Understand practical implementation details and best practices  
- **AI Enthusiasts**: Get hands-on experience with state-of-the-art fine-tuning techniques
- **Domain Experts**: Adapt language models to your specific field (medical, legal, scientific, etc.)

**No prior fine-tuning experience required!** We'll walk through everything step-by-step.

---

## What You'll Learn

In this tutorial, we'll fine-tune Gemma 3 270M on a custom radiobiology dataset. You'll master:

1. **Data Creation**: Generate high-quality training data from scratch using LLM-DATA-Generator
2. **Model Setup**: Load and configure Gemma 3 models with KerasHub
3. **LoRA Fine-Tuning**: Apply parameter-efficient fine-tuning techniques
4. **Training**: Optimize hyperparameters and monitor training progress
5. **Evaluation**: Test model performance before and after fine-tuning
6. **Deployment**: Save and use your fine-tuned model for inference

---

## Hardware Requirements: A CPU-First Approach

### This Tutorial Uses CPU

We'll be fine-tuning Gemma 3 270M on CPU to make this tutorial **accessible to everyone**!

- ✅ No expensive GPU required
- ✅ Runs on any laptop or desktop
- ✅ Perfect for learning and experimentation
- ✅ Same code works everywhere

### Want Faster Training? Use the Same Code!

The beauty of this tutorial is that **the exact same code** can run on different hardware:

**Option 1: CPU (What We're Using)**
- Works on any device
- No special setup required
- Great for learning
- Training time: Slower but totally doable for Gemma 270M

**Option 2: GPU (Same Code, Faster!)**
```bash
pip install keras-hub[jax] jax[cuda12]  # For NVIDIA GPUs
```
- No code changes needed — it automatically detects your GPU!
- Training time: Much faster (minutes instead of hours)

**Option 3: Google Colab (Free GPU!)**
- Upload this notebook to [Google Colab](https://colab.research.google.com/)
- Runtime → Change runtime type → GPU (T4 with 15GB VRAM)
- Same code, zero installation, free GPU access!

---

## A Word from Your CPU

![CPU saying fine-tuning is not my job](cpu-usage-training.png)

*Your CPU when you ask it to fine-tune a language model* 😅

But guess what? With Gemma 270M and modern optimization techniques like LoRA, your CPU **can actually do it**! It might take a coffee break (or two ☕☕), but it gets the job done.

---

## The Complete Workflow

Here's what we'll accomplish:

```
📚 Step 1: Data Creation
   ├─ Source: Radiobiology textbook (PDF)
   ├─ Tool: LLM-DATA-Generator
   └─ Output: 832 Q&A pairs (CSV)
   
⬇️

🔧 Step 2: Environment Setup
   ├─ Install Keras & KerasHub
   ├─ Configure Kaggle credentials
   └─ Set JAX backend for optimal performance
   
⬇️

🤖 Step 3: Model Loading
   ├─ Download Gemma 3 270M from Kaggle
   ├─ Load pre-trained weights
   └─ Test baseline performance
   
⬇️

🎯 Step 4: LoRA Configuration
   ├─ Enable LoRA (rank=8)
   ├─ Freeze base model weights
   └─ Configure trainable parameters
   
⬇️

📈 Step 5: Training
   ├─ Prepare dataset
   ├─ Set hyperparameters
   └─ Train the model
   
⬇️

✅ Step 6: Evaluation & Inference
   ├─ Compare before/after performance
   ├─ Test on domain-specific questions
   └─ Save fine-tuned model
```

---

## Time Commitment

- **Reading & Understanding**: ~30 minutes
- **Running the Code** (with GPU): ~45-60 minutes
- **Total**: ~1.5-2 hours for complete mastery

---

## What's Included

- ✅ Complete, runnable code with detailed explanations
- ✅ Pre-generated dataset (832 Q&A pairs)
- ✅ Step-by-step instructions with visual aids
- ✅ Best practices and optimization tips
- ✅ Troubleshooting guidance
- ✅ Links to additional resources

---

## The Fine-Tuning Architecture

![Fine-Tuning Architecture](lora-architecture.png)

Understanding the architecture is crucial before diving into implementation. The diagram above shows our complete fine-tuning pipeline — from raw data to a specialized model.

---

## Step 1: Data Generation — Creating Your Training Dataset

### The Challenge

How long would it take to manually create 1,000 high-quality Q&A pairs from a textbook? Probably weeks! That's why we automate it.

### Solution: LLM-DATA-Generator

We use the **LLM-DATA-Generator** tool to automatically generate training data from the Radiobiology textbook.

**🔗 Tool Repository**: [ElHadheqMind/LLM-DATA-Generator](https://github.com/ElHadheqMind/LLM-DATA-Generator)

### How It Works

```
📖 Input: Radiobiology Textbook (PDF)
    ↓
🤖 LLM-DATA-Generator Magic
    ↓
📊 Output: 832 Q&A Pairs (CSV)
    ↓
✅ Ready for Fine-Tuning!
```

**Time saved**: ~2 weeks of manual work → 30 minutes automated!

### Dataset Generation in Action

![Dataset Generation Demo](llm-data-generator-demo.gif)

The process covers:
1. **Setup**: Installing and configuring LLM-DATA-Generator
2. **API Configuration**: Setting up LLM API credentials (OpenAI/Anthropic/Google)
3. **Document Upload**: Loading the source textbook (PDF/DOCX)
4. **Parameter Tuning**: Configuring generation settings (question count, difficulty, format)
5. **Generation**: Running the automated Q&A extraction process
6. **Export**: Reviewing and exporting the final CSV dataset

### Dataset Structure

| Column | Description | Example |
|--------|-------------|----------|
| **Question** | Domain-specific question | "What is the primary mechanism by which radiation damages DNA?" |
| **Answer** | Comprehensive answer with context | "Radiation damages DNA primarily through..." |
| **Content** | Source text from the textbook | Original section from the book |

---

## Step 2: Environment Setup

### Prerequisites

Before starting, complete the [Gemma setup](https://ai.google.dev/gemma/docs/setup) instructions:

1. Get access to Gemma on [kaggle.com](https://kaggle.com)
2. Optional: GPU resources (RTX 4060 8GB or similar) for faster training
3. Generate and configure a Kaggle username and API key

### Kaggle Configuration

Gemma models are hosted on Kaggle. You'll need API credentials to download them programmatically.

#### Create Kaggle Account & Accept Terms

1. **Sign up**: [kaggle.com/account/login](https://www.kaggle.com/account/login?phase=startRegisterTab)
2. **Accept Gemma license**: [kaggle.com/models/google/gemma-3](https://www.kaggle.com/models/google/gemma-3)
   Click any variant → "Request Access" → Accept terms

**📚 Reference**: [Kaggle Models Documentation](https://www.kaggle.com/docs/models)

#### Generate API Token

1. Profile → **Settings** → Scroll to **API** section
2. Click **"Create New Token"** → Downloads `kaggle.json`

**File structure**:
```json
{"username": "your_username", "key": "abc123..."}
```

**🔒 Security**: This file grants full API access. Never commit to Git or share publicly.

### Load Kaggle Credentials

```python
import os
import json

# Load credentials from kaggle.json
with open('kaggle.json', 'r') as f:
    kaggle_creds = json.load(f)

os.environ["KAGGLE_USERNAME"] = kaggle_creds['username']
os.environ["KAGGLE_KEY"] = kaggle_creds['key']

print("✅ Kaggle credentials loaded from kaggle.json")
print(f"   Username: {kaggle_creds['username']}")
```

**Output:**
```
✅ Kaggle credentials loaded from kaggle.json
   Username: mezzihoussem
```

### Install Required Packages

```python
!pip install -q -U keras-hub
!pip install -q -U keras

print("✅ Packages installed successfully!")
```

**Output:**
```
✅ Packages installed successfully!
```

We're installing:
- **Keras 3**: Multi-framework deep learning API (supports JAX, TensorFlow, PyTorch)
- **KerasHub**: Pre-trained models and utilities for NLP tasks, including Gemma

### Select Keras Backend

**Keras 3** is a multi-framework API that supports three backends:

| Backend | Pros | Cons | Best For |
|---------|------|------|----------|
| **JAX** | Fastest training, XLA compilation, best GPU utilization | Newer ecosystem | Production training |
| **TensorFlow** | Mature ecosystem, TensorFlow Serving | Slower than JAX | Deployment |
| **PyTorch** | Popular, extensive community | Slower than JAX | Research |

**For this tutorial**, we use **JAX** because:
- Best performance for LoRA fine-tuning
- Optimized for NVIDIA GPUs (including RTX 4060)
- Recommended by Google for Gemma models
- Efficient memory management for 8GB VRAM

**Reference**: [Keras 3 Multi-Backend Documentation](https://keras.io/keras_3/)

```python
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

print("✅ Backend configured: JAX")
```

**Output:**
```
✅ Backend configured: JAX
```

### Import Libraries

```python
import keras
import keras_hub
import pandas as pd
import numpy as np
import sys

print(f"✅ Keras version: {keras.__version__}")
print(f"✅ KerasHub version: {keras_hub.__version__}")
print(f"✅ Backend: {keras.backend.backend()}")
```

**Output:**
```
✅ Keras version: 3.12.0
✅ KerasHub version: 0.23.0
✅ Backend: jax
```

### Verify Package Versions

This tutorial was tested with:
- **Keras**: 3.12.0
- **KerasHub**: 0.23.0
- **TensorFlow**: 2.19.0
- **JAX**: 0.7.2
- **Python**: 3.12.12

```python
import tensorflow as tf
import jax

print("="*60)
print("PACKAGE VERSIONS")
print("="*60)
print(f"Keras version: {keras.__version__}")
print(f"Keras-Hub version: {keras_hub.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"JAX version: {jax.__version__}")
print(f"Python version: {sys.version}")
print("="*60)
```

**Output:**
```
============================================================
PACKAGE VERSIONS
============================================================
Keras version: 3.12.0
Keras-Hub version: 0.23.0
TensorFlow version: 2.19.0
JAX version: 0.7.2
TensorFlow Text version: 2.19.0
Python version: 3.12.3 (main, Aug 14 2025, 17:47:21) [GCC 13.3.0]
============================================================

✅ All packages loaded successfully!
```

---

## Step 3: Load Gemma 3 Model

### What is a Causal Language Model?

A **Causal Language Model (CLM)** predicts the next token based on previous tokens in a sequence. This is the foundation of text generation:

```
Input:  "The radiation dose is"
Output: "measured in Gray (Gy)"
```

### Available Gemma 3 Presets

KerasHub provides several pre-configured Gemma 3 models:

| Preset Name | Size | Context | Type | Memory |
|-------------|------|---------|------|--------|
| `gemma3_instruct_270m` | 270M | 8K | Text-only | ~0.5 GB |
| `gemma3_instruct_1b` | 1B | 32K | Text-only | ~1.5 GB |
| `gemma3_instruct_4b` | 4B | 128K | Multimodal | ~6.4 GB |
| `gemma3_instruct_12b` | 12B | 128K | Multimodal | ~20 GB |
| `gemma3_instruct_27b` | 27B | 128K | Multimodal | ~46 GB |

**For this tutorial**, we use `gemma3_instruct_270m` — optimized for instruction following and fine-tuning on consumer CPUs and GPUs.

**Browse all models**: [Gemma 3 on Kaggle](https://www.kaggle.com/models/keras/gemma3)

### Loading Process

The `from_preset()` method will:
1. Download the model weights from Kaggle (~1.5 GB)
2. Load the tokenizer (262K vocabulary)
3. Initialize the model architecture
4. Load pre-trained weights

**Note**: First-time download may take 2-5 minutes depending on your connection.

```python
print("📥 Loading Gemma 3 270m Instruct model...")
print("⏳ This may take 2-5 minutes on first run...\n")

gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_270m")

print("\n✅ Model loaded successfully!")
```

**Output:**
```
📥 Loading Gemma 3 270m Instruct model...
⏳ This may take 2-5 minutes on first run...


✅ Model loaded successfully!
```

---

## Step 4: Inference Before Fine-Tuning

Before fine-tuning, let's test how Gemma 3 270M responds to different prompt formats. **Gemma 3 270M is highly sensitive to prompt engineering** — the way you structure your prompt significantly impacts response quality.

### Sampling Strategy

We use **`keras_hub.samplers.TopPSampler`** for controlled text generation:
- **Top-P (nucleus) sampling**: Selects from the smallest set of tokens whose cumulative probability exceeds `p`
- **Top-K filtering**: Limits selection to the `k` most likely tokens
- **Parameters**: `p=0.3, k=5, seed=2` for reproducible, focused outputs

### Prompt Template Formats

We'll test three approaches:

**1. Custom Template (Simple)**
```python
"System Instruction:\n{system_instruction}\n\nQuestion:\n{instruction}\n\nResponse:\n{response}"
```

**2. Gemma Official Chat Template**
```python
"<start_of_turn>user\n{content}\n<end_of_turn>\n<start_of_turn>model\n"
```

### Test 1: No Chat Template, No System Instruction

**Approach**: Simple instruction-response format without system context.

```python
template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

prompt = template.format(
    instruction="What is the double-helix molecule primarily responsible for heredity and the synthesis of proteins?",
    response="",
)
sampler = keras_hub.samplers.TopPSampler(p=0.3, k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=256))
```

**Output:**
```
Instruction:
What is the double-helix molecule primarily responsible for heredity and the synthesis of proteins?

Response:
The double-helix molecule primarily responsible for heredity and the synthesis of proteins is a **protein**.
<end_of_turn>
```

**Result**: ❌ **WRONG!** The model says "protein" instead of "DNA". Without system instruction, Gemma 3 270M struggles with basic questions.

### Test 2: Custom Template + System Instruction

**Approach**: Add system instruction to guide the model's behavior.

```python
SYSTEM_INSTRUCTION = "You are a concise and expert assistant in Radiobiology. Provide accurate, clear, and relevant answers."
instruction_text = "What is the double-helix molecule primarily responsible for heredity and the synthesis of proteins?"
template = "System Instruction:\n{system_instruction}\n\nQuestion:\n{instruction}\n\nResponse:\n{response}"
prompt = template.format(
    system_instruction=SYSTEM_INSTRUCTION,
    instruction=instruction_text,
    response="",
)

sampler = keras_hub.samplers.TopPSampler(p=0.3, k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=256))
```

**Output:**
```
System Instruction:
You are a concise and expert assistant in Radiobiology. Provide accurate, clear, and relevant answers.

Question:
What is the double-helix molecule primarily responsible for heredity and the synthesis of proteins?

Response:
The double-helix molecule primarily responsible for heredity and the synthesis of proteins is the **DNA**.
<end_of_turn>
```

**Result**: ✅ **CORRECT!** The model now correctly answers "DNA". System instruction provides crucial context.

### Test 3: Gemma Official Chat Template

**Approach**: Use Gemma's official chat format with turn delimiters.

```python
START_TURN_USER = "<start_of_turn>user\n"
END_TURN = "<end_of_turn>\n"
START_TURN_MODEL = "<start_of_turn>model\n"

SYSTEM_INSTRUCTION = "You are a concise and expert assistant in Radiobiology. Provide accurate, clear, and relevant answers."
USER_QUESTION = "What is the double-helix molecule primarily responsible for heredity and the synthesis of proteins?"

user_content = f"{SYSTEM_INSTRUCTION}Question:\n{USER_QUESTION}"

prompt = (
    f"{START_TURN_USER}"
    f"{user_content}\n"
    f"{END_TURN}"
    f"{START_TURN_MODEL}"
)

sampler = keras_hub.samplers.TopPSampler(p=0.3, k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=256))
```

**Output:**
```
<start_of_turn>user
You are a concise and expert assistant in Radiobiology. Provide accurate, clear, and relevant answers.Question:
What is the double-helix molecule primarily responsible for heredity and the synthesis of proteins?
<end_of_turn>
<start_of_turn>model
The double-helix molecule primarily responsible for heredity and the synthesis of proteins is **DNA**.
<end_of_turn>
```

**Result**: ✅ **CORRECT!** Gemma chat template also produces accurate response. Both custom and official templates work when system instruction is included.

### Test 4: Custom Template — Looping Issue

Testing with a different question reveals an important issue:

```python
instruction_text = "Who Discovered the Structure of DNA?"
# ... same template setup ...
print(gemma_lm.generate(prompt, max_length=256))
```

**Output:**
```
System Instruction:
You are a concise and expert assistant in Radiobiology. Provide accurate, clear, and relevant answers.

Question:
Who Discovered the Structure of DNA?

Response:
The discovery of the structure of DNA was made by the geneticist, **Watson and Crick**.

Question:
What is the purpose of the DNA double helix?

Response:
The DNA double helix is a structure that allows for the precise replication of genetic information...

Question:
What is the function of the DNA double helix?

Response:
The DNA double helix is a structural framework that provides the stability and organization...
```

**Result**: ⚠️ **LOOPING!** The model generates repetitive Q&A pairs instead of stopping. Custom template can cause instability with Gemma 3 270M.

### Test 5: Gemma Chat Template — Stable Output

Same question with Gemma's official template:

**Output:**
```
<start_of_turn>
You are a concise and expert assistant in Radiobiology. Provide accurate, clear, and relevant answers.Question:
Who Discovered the Structure of DNA?
<end_of_turn>
<start_of_turn>Response:
The discovery of the structure of DNA was made by **Watson and Crick** in 1953.<end_of_turn>
```

**Result**: ✅ **STABLE & CORRECT!** Gemma chat template produces concise, accurate response without looping.

### Summary: Prompt Engineering Matters

| Template Type | System Instruction | Result | Issue |
|---------------|-------------------|--------|-------|
| Simple | ❌ No | ❌ Wrong answer | Hallucination ("protein" instead of "DNA") |
| Custom | ✅ Yes | ✅ Correct | Works but can cause looping |
| Gemma Official | ✅ Yes | ✅ Correct | Stable, no looping |

**Recommendations**:

1. **For Gemma 3 270M**: Use **Gemma official chat template** for best stability
2. **Always include system instructions** — critical for accuracy
3. **Custom templates work** but may cause repetitive outputs
4. **Larger models (1B+)** are less sensitive to prompt format

---

## Step 5: LoRA Fine-Tuning

This section shows you how to do fine-tuning using the Low Rank Adaptation (LoRA) tuning technique. This approach allows you to change the behavior of Gemma models using fewer compute resources.

### Load Custom Radiobiology Dataset

```python
import pandas as pd

df = pd.read_csv("data.csv")
prompts = df["Question"].tolist()
responses = df["Answer"].tolist()
SYSTEM_INSTRUCTION = "You are a concise and expert assistant in Radiobiology. Provide accurate, clear, and relevant answers.\n"

updated_prompts = [SYSTEM_INSTRUCTION + p for p in prompts]
data = {
    "prompts": prompts,
    "responses": responses
}

print("\n--- Example of the First Template-Based Prompt (Input) ---\n")
print(data["prompts"][5])
print("\n--- Example of the Corresponding Response (Target) ---\n")
print(data["responses"][5])
```

**Output:**
```
--- Example of the First Template-Based Prompt (Input) ---

Describe the process of DNA damage repair following ionizing radiation exposure, focusing on at least three distinct mechanisms.

--- Example of the Corresponding Response (Target) ---

Following ionizing radiation exposure, DNA undergoes significant damage, primarily in the form of single-strand breaks (SSBs) and double-strand breaks (DSBs). Cellular response involves a complex network of DNA repair pathways to mitigate this damage. One key mechanism is Non-Homologous End Joining (NHEJ), which directly ligates broken DNA ends without requiring a homologous template. This process is error-prone, often resulting in small insertions or deletions. Another crucial pathway is Homologous Recombination (HR), which utilizes the undamaged sister chromatid as a template to accurately repair DSBs. HR requires stalled replication forks and proceeds through multiple steps including strand invasion and resolution. Finally, Base Excision Repair (BER) addresses damage to individual DNA bases caused by radiation-induced oxidation or alkylation. BER involves recognition of damaged bases by glycosylases, followed by removal of the damaged base and subsequent replacement with an appropriate nucleotide using DNA polymerase. These three mechanisms – NHEJ, HR, and BER – represent critical components of the cellular response to ionizing radiation damage, each contributing uniquely to maintaining genomic stability. Reference: Molecular Radiation Biology, Section 3.2 – DNA Damage Repair Mechanisms.
```

#### Dataset Overview

- **832 question-answer pairs** from "Molecular Radiation Biology" textbook
- **Domain**: Radiobiology, radiation physics, medical applications
- **Format**: CSV with columns: `Question`, `Answer`, `Content`

#### Chat Template Strategy for Fine-Tuning

**You can fine-tune with any chat template!** In this tutorial, we will **not use any specific chat template** to keep the fine-tuning general and flexible. This allows you to use the fine-tuned model with **any chat template** after training.

**Option 1: Custom Template (Used in this tutorial)**
```python
CUSTOM_TEMPLATE = (
    "System Instruction:\n{system_instruction}\n\n"
    "Question:\n{instruction}\n\n"
    "Response:\n{response}"
)
```

**Option 2: Gemma Official Chat Template**
```python
GEMMA_TEMPLATE = (
    "<start_of_turn>user\n"
    "{system_instruction}\n\n"
    "{instruction}<end_of_turn>\n"
    "<start_of_turn>model\n"
    "{response}<end_of_turn>"
)
```

**💡 Tip**: Experiment with both templates during fine-tuning to see which produces better results for your specific domain!

---

### Configure LoRA Fine-Tuning

#### Understanding LoRA Rank

The **rank (r)** parameter is crucial — it determines the dimensionality of the low-rank matrices:

**Mathematical Perspective**:
- Original weight matrix: **W** (d × d) — frozen
- LoRA matrices: **A** (d × r) and **B** (r × d) — trainable
- Updated weight: **W' = W + B × A**

**Practical Impact**:

| Rank | Use Case | Trade-offs |
|------|----------|------------|
| **Lower (4, 8)** | Simple tasks | Faster training, less memory |
| **Medium (16, 32)** | Most domain adaptation | Balanced performance |
| **Higher (64, 128+)** | Complex tasks | More expressive, requires more resources |

#### Our Choice: Rank = 8

For this tutorial, we use **rank=8** because:
- Good balance between performance and efficiency
- Works well on RTX 4060 8GB
- Sufficient for domain-specific Q&A adaptation
- Recommended starting point for most tasks

**Reference**: [LoRA Paper - Section 4.2](https://arxiv.org/abs/2106.09685)

### Enable LoRA

```python
# Enable LoRA with rank=8
gemma_lm.backbone.enable_lora(rank=8)
```

The `enable_lora()` method has other parameters we're keeping at their default values:

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `rank` | **8** (we set this!) | Size of the LoRA factorization matrices |
| `target_layer_names` | `None` (auto) | Which layers get LoRA (defaults to query & value layers) |

When `target_layer_names=None`, KerasHub automatically targets:
- `"query_dense"`, `"value_dense"`, `"query"`, `"value"`

These are the attention mechanism's query and value projection layers — the most important parts for fine-tuning!

**📚 Source code**: [KerasHub `enable_lora()` source code](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/backbone.py#L191)

### Check Model Summary

```python
gemma_lm.summary()
```

**Output:**
```
Preprocessor: "gemma3_causal_lm_preprocessor"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                                                  ┃                                   Config ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ gemma3_tokenizer (Gemma3Tokenizer)                            │                      Vocab size: 262,144 │
└───────────────────────────────────────────────────────────────┴──────────────────────────────────────────┘

Model: "gemma3_causal_lm"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ padding_mask (InputLayer)     │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_ids (InputLayer)        │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ gemma3_backbone               │ (None, None, 640)         │     268,632,704 │ padding_mask[0][0],        │
│ (Gemma3Backbone)              │                           │                 │ token_ids[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_embedding               │ (None, None, 262144)      │     167,772,160 │ gemma3_backbone[0][0]      │
│ (ReversibleEmbedding)         │                           │                 │                            │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘

 Total params: 268,632,704 (1.00 GB)
 Trainable params: 534,528 (2.04 MB)
 Non-trainable params: 268,098,176 (1022.71 MB)
```

Notice that enabling LoRA reduces the number of trainable parameters significantly:
- **Total params**: 268,632,704 (1.00 GB)
- **Trainable params**: 534,528 (2.04 MB)
- **Non-trainable params**: 268,098,176 (1022.71 MB)

**That's only 0.2% of parameters being trained!** This is the magic of LoRA.

---

## Step 6: Configure Training Parameters

### Sequence Length

**Setting**: `1024 tokens` (~750 words)

This limits how much text the model processes at once. Think of it as the model's "attention span."

- **Too short (256)**: Model can't see enough context
- **Too long (4096)**: Your GPU goes out of memory
- **Just right (1024)**: Fits our 8GB GPU perfectly

```python
gemma_lm.preprocessor.sequence_length = 1024
```

### Optimizer: AdamW

Think of the optimizer as a **GPS for finding the best model weights**. AdamW is smart, efficient, and knows when to slow down.

**We're using 2 out of 16 available parameters**:

#### 1. `learning_rate = 5e-5` (0.00005)

How big of a step we take when updating weights:
- **Too high (1e-3)**: Model overshoots the target
- **Too low (1e-6)**: Training takes forever
- **Just right (5e-5)**: Steady progress!

#### 2. `weight_decay = 0.01`

Regularization that prevents overfitting (memorizing training data). Like telling a student "understand the concept, don't just memorize!"

```python
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Exclude bias and scale parameters from weight decay (best practice)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
```

**Why exclude bias & scale?** These parameters are like the "seasoning" in our model — they're already small and well-behaved. Decaying them can hurt performance.

**📚 Reference**: [Keras AdamW Optimizer Documentation](https://keras.io/api/optimizers/adamw/)

### Compile the Model

```python
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

**Components**:

1. **Loss Function: `SparseCategoricalCrossentropy`**
   - **"Sparse"** = targets are integers (token IDs), not one-hot vectors
   - **"Categorical"** = choosing from 262,144 possible tokens
   - **`from_logits=True`** = Model outputs raw scores, not probabilities

2. **Metrics: `SparseCategoricalAccuracy`**
   - Percentage of tokens predicted correctly
   - Easy to understand — "Did we guess the right word?"

### Mixed Precision (Optional)

For NVIDIA GPUs, you can enable mixed precision to speed up training:

```python
keras.mixed_precision.set_global_policy('mixed_bfloat16')
```

**Fun fact**: Even though this is labeled for GPUs, your CPU can accept it too! Mixed precision works on CPUs, though the speedup benefits are primarily seen on modern GPUs with tensor cores.

---

## Step 7: Run the Fine-Tuning Process

Now for the main event — actually training our model!

### Understanding the `fit()` Parameters

```python
gemma_lm.fit(data, epochs=3, batch_size=1)
```

**1. epochs = 3**

How many times the model sees the ENTIRE dataset:
- 1 epoch = Read it once
- 3 epochs = Read it three times
- 10 epochs = You're basically memorizing it!

**Why 3?** For LoRA fine-tuning on domain-specific data, 3 epochs provides a good balance. You'll see loss decrease: Epoch 1 (0.67) → Epoch 2 (0.63) → Epoch 3 (0.62)

**2. batch_size = 1**

How many examples the model processes before updating weights:
- batch_size=1: Grade one paper, update your answer key, repeat
- batch_size=32: Grade 32 papers, THEN update your answer key

**Why 1?** Memory constraints on our 8GB GPU. Larger batches need more VRAM.

### Training Time Expectations

- **RTX 4060 8GB**: ~15-30 minutes for 832 examples × 3 epochs
- **RTX 3090 24GB**: ~10-15 minutes (with larger batch_size)
- **CPU only**: Hours (grab several coffees ☕☕☕)

### What to Watch During Training

- **Loss**: Should decrease across epochs (lower is better!)
- **Accuracy**: Should increase across epochs (higher is better!)
- **GPU Memory**: Should stay under 8GB (check with `nvidia-smi`)

### Training Results

```
Epoch 1/3
831/831 ━━━━━━━━━━━━━━━━━━━━ 2542s 3s/step - loss: 0.6714 - sparse_categorical_accuracy: 0.4577
Epoch 2/3
831/831 ━━━━━━━━━━━━━━━━━━━━ 2538s 3s/step - loss: 0.6310 - sparse_categorical_accuracy: 0.4729
Epoch 3/3
831/831 ━━━━━━━━━━━━━━━━━━━━ 2532s 3s/step - loss: 0.6178 - sparse_categorical_accuracy: 0.4801
```

The loss is decreasing and accuracy is increasing — our model is learning!

---

## Step 8: Inference After Fine-Tuning

Let's see how our fine-tuned model performs!

### Test 1: Gemma Official Chat Template

```python
START_TURN_USER = "<start_of_turn>\n"
END_TURN = "<end_of_turn>\n"
START_TURN_MODEL = "<start_of_turn>Response:\n"

SYSTEM_INSTRUCTION = "You are a concise and expert assistant in Radiobiology. Provide accurate, clear, and relevant answers."
USER_QUESTION = "Describe the process of DNA repair in bacteria, outlining at least three distinct mechanisms and explaining the specific damage each addresses."

user_content = f"{SYSTEM_INSTRUCTION}Question:\n{USER_QUESTION}"

prompt = (
    f"{START_TURN_USER}"
    f"{user_content}\n"
    f"{END_TURN}"
    f"{START_TURN_MODEL}"
)

sampler = keras_hub.samplers.TopPSampler(p=0.1, k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=512))
```

**Output:**
```
<start_of_turn>
You are a concise and expert assistant in Radiobiology. Provide accurate, clear, and relevant answers.Question:
Describe the process of DNA repair in bacteria, outlining at least three distinct mechanisms and explaining the specific damage each addresses.
<end_of_turn>
<start_of_turn>Response:
DNA repair is a crucial process in bacterial cells that protects the integrity of the genetic code by repairing damaged DNA strands. It primarily occurs through two main mechanisms:

1. **Base excision and circular DNA strand breakage:** This mechanism involves the removal of a portion of the double-strand overhangs (breaks) on the DNA backbone. This process is triggered by mutations that cause mutations in the bacterial genome. The repair process is often triggered by the addition of a DNA repair enzyme, such as BRCA1 or BRCA2, which specifically targets the damaged base pairs.

2. **Circular DNA strand repair:** This mechanism involves the removal of a portion of the circular DNA strand, often referred to as the "replication problem." This occurs when a mutation occurs that causes the circular DNA to break, leading to a new, identical strand. The repair process is triggered by the addition of a DNA repair enzyme, such as BRCA1 or BRCA2, which specifically targets the circular DNA.

3. **Non-homologous strand breakage:** This mechanism involves the removal of a portion of the non-homologous strand (the strand that is not part of the original DNA sequence). This occurs when a mutation occurs that causes the non-homologous strand to break, leading to a new, identical strand.

In summary, DNA repair utilizes a combination of enzymes and mechanisms to correct errors in DNA replication and repair the double-strand break that occurs during replication.<end_of_turn>
```

**Result**: The model provides a detailed response about DNA repair mechanisms including base excision, circular DNA strand repair, and non-homologous strand breakage.

### Test 2: Custom Template

```python
SYSTEM_INSTRUCTION = "You are a concise and expert assistant in Radiobiology. Provide accurate, clear, and relevant answers."
instruction_text = "Describe the process of DNA repair in bacteria, outlining at least three distinct mechanisms and explaining the specific damage each addresses."
template = "System Instruction:\n{system_instruction}\n\nQuestion:\n{instruction}\n\nResponse:\n{response}"
prompt = template.format(
    system_instruction=SYSTEM_INSTRUCTION,
    instruction=instruction_text,
    response="",
)

sampler = keras_hub.samplers.TopPSampler(p=0.1, k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=512))
```

**Output:**
```
System Instruction:
You are a concise and expert assistant in Radiobiology. Provide accurate, clear, and relevant answers.

Question:
Describe the process of DNA repair in bacteria, outlining at least three distinct mechanisms and explaining the specific damage each addresses.

Response:
DNA repair in bacteria is a complex process that occurs in the cell's genetic material. This process is essential for maintaining genome integrity and ensuring the accurate replication of the genetic code. Here's a breakdown of three distinct mechanisms:

1. **Base Excision Repair (BER):** This mechanism is responsible for repairing minor DNA lesions such as damaged bases, single-strand breaks, and nucleotide excision repair (NER)-related DNA lesions. BER uses a specialized enzyme called uracil-DNA glycosylase to remove the damaged base and repair the double strand.

2. **Non-Homologous End Joining (NHEJ):** This mechanism is responsible for repairing DNA double-strand breaks (DSBs). NHEJ uses a specialized enzyme called the Ku heterodimer, which recognizes the damaged ends of the DNA and facilitates the recruitment of the other repair machinery.

3. **Homologous Recombination (HR):** This mechanism is responsible for repairing DNA DSBs in S-phase cells. HR utilizes a specialized enzyme called the RecA protein, which catalyzes the strand invasion step, allowing for the repair of the DSB.

Reference: [MolecularRadiationBiology, Section 10.1.1 – DNA Repair Mechanisms]<end_of_turn>
```

**Result**: The model provides a detailed response AND includes a reference citation at the end: `Reference: [MolecularRadiationBiology, Section 10.1.1 – DNA Repair Mechanisms]`

### Comparison: Which Template Performed Better?

| Template | Result Quality | Key Observation |
|----------|---------------|------------------|
| **Gemma Official** | Detailed | Provides mechanisms but no source reference |
| **Custom Template** | Detailed + **Reference** | Includes reference citation from training data |

**⚠️ Important Note**: Both responses are **detailed but not necessarily accurate** from a scientific standpoint. The model learned patterns from the training data, but accuracy requires domain expert validation.

**Key Insight**: The **Custom Template** gave us the **reference citation** at the end, suggesting it's **more closely converged to the training data format**. Our training data included references, and the custom template successfully learned to reproduce this pattern!

This demonstrates that:
- ✅ The model **learned the domain knowledge** (radiobiology concepts)
- ✅ The model **learned the output format** (including references)
- ✅ Custom templates can preserve training data patterns better
- ⚠️ **Detailed ≠ Accurate** — Always validate outputs with domain experts!

**Recommendation**: If your training data has specific formatting (like references), using a **custom template that matches your training format** may yield better results!

---

## Step 9: Save Your Fine-Tuned Model

Now that we've fine-tuned our model, let's save it for later use!

```python
preset_dir = "./radiobiology_gemma3_model"
gemma_lm.save_to_preset(preset_dir)
```

**What gets saved**:
- ✅ **Model weights** (base model + LoRA adapters merged)
- ✅ **Tokenizer** (vocabulary and configuration)
- ✅ **Preprocessor** (text processing settings)
- ✅ **Configuration files** (all model settings)

**Why save as a preset?**
- 📦 **Self-contained**: Everything in one directory
- 🚀 **Easy to share**: Upload to Kaggle Models or Hugging Face
- 🔄 **Easy to reload**: One line of code to load later

---

## Step 10: Load Your Fine-Tuned Model Later

Want to use your fine-tuned model in a new session? Just load it from the preset directory!

```python
gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("./radiobiology_gemma3_model")
```

**This is useful when**:
- 🔁 **Restarting your notebook** — No need to fine-tune again!
- 🚀 **Deploying to production** — Load the model in your application
- 🤝 **Sharing with others** — They can load your fine-tuned model easily
- 💻 **Moving to a different machine** — Just copy the directory and load

---

## Improving Fine-Tune Results: Level Up Your Model

Our results are already impressive, but there's always room for improvement!

### Current Configuration Summary

| Parameter | Current Value | Purpose |
|-----------|---------------|----------|
| **Model** | `gemma3_instruct_270m` | Base model (270M parameters) |
| **LoRA Rank** | `rank=8` | Adapter expressiveness |
| **Epochs** | `epochs=3` | Training passes through data |
| **Batch Size** | `batch_size=1` | Examples per weight update |
| **Learning Rate** | `5e-5` (0.00005) | Step size for weight updates |
| **Weight Decay** | `0.01` | Regularization strength |
| **Dataset Size** | 832 Q&A pairs | Training examples |
| **Sequence Length** | 1024 tokens | Max input/output length |

### Tuning Strategies (From Easiest to Most Advanced)

#### 1. Increase Training Epochs

- **Current**: `epochs=3`
- **Try**: `epochs=5` or `epochs=7` for better convergence
- **Impact**: Model sees examples multiple times, learns patterns deeper
- **⚠️ Watch out**: Too many epochs (10+) can cause overfitting

#### 2. Increase LoRA Rank

- **Current**: `rank=8`
- **Try**: `rank=16` or `rank=32` for more capacity
- **Impact**: More expressive adapters, can learn more complex patterns
- **⚠️ Watch out**: Higher rank = more memory usage and slower training

#### 3. Adjust Learning Rate

- **Current**: `learning_rate=5e-5`
- **Try higher**: `1e-4` for faster convergence (riskier)
- **Try lower**: `1e-5` for more stable training (slower)

#### 4. Tune Weight Decay

- **Current**: `weight_decay=0.01`
- **If overfitting**: Increase to `0.05` or `0.1`
- **If underfitting**: Decrease to `0.001` or `0.005`

#### 5. Increase Batch Size

- **Current**: `batch_size=1`
- **If you have 16GB+ VRAM**: Try `batch_size=4` or `batch_size=8`
- **Impact**: 2-4x faster training, more stable gradient updates

#### 6. Expand Your Dataset

- **Current**: 832 Q&A pairs
- **Target**: 2000-5000 examples for robust domain adaptation
- **💡 Tip**: Use LLM-DATA-Generator to create more training data!

#### 7. Increase Sequence Length

- **Current**: `sequence_length=1024`
- **Try**: `sequence_length=2048` for longer contexts (needs more VRAM)

### Recommended Improvement Path

**For Better Results (Easy)**:
1. ✅ Increase epochs to 5-7
2. ✅ Increase LoRA rank to 16
3. ✅ Add more training data (aim for 2000+ examples)

**For Faster Training (If you have VRAM)**:
1. ✅ Increase batch_size to 4 or 8
2. ✅ Slightly increase learning_rate to 1e-4

**For Production Quality**:
1. ✅ All of the above
2. ✅ Implement validation split to monitor overfitting
3. ✅ Use learning rate scheduling (warmup + decay)
4. ✅ Test multiple LoRA ranks and pick the best

### Pro Tips

1. **Start small**: Try one change at a time to see what works
2. **Monitor GPU memory**: Use `nvidia-smi` to check you're not hitting limits
3. **Validate results**: Test on questions NOT in your training data
4. **Save checkpoints**: Save your model after each experiment
5. **Track metrics**: Keep a spreadsheet of what works and what doesn't
6. **Watch the loss curve**: Loss should decrease smoothly — if it spikes, reduce learning rate
7. **Compare templates**: Test both Gemma and Custom templates after each change

**Remember**: The best configuration depends on your specific use case, dataset, and hardware!

---

## Conclusion

Congratulations! You've successfully fine-tuned Gemma 3 270M using LoRA on a custom radiobiology dataset. Here's what we accomplished:

1. **Created training data** using LLM-DATA-Generator from a textbook
2. **Set up the environment** with Keras 3 and JAX backend
3. **Loaded and tested** the base Gemma 3 model
4. **Enabled LoRA** to reduce trainable parameters by 99.8%
5. **Trained the model** on domain-specific Q&A pairs
6. **Evaluated the results** using different prompt templates
7. **Saved the model** for future use

The key takeaways:

- **LoRA makes fine-tuning accessible** — even on CPU!
- **Prompt engineering matters** — especially for smaller models
- **Custom templates can preserve training patterns** better than generic ones
- **Always validate with domain experts** — detailed doesn't mean accurate

### Next Steps

- **Expand your dataset** with more textbooks or papers
- **Experiment with larger models** (1B, 4B) if you have the hardware
- **Try different LoRA ranks** to find the optimal balance
- **Deploy your model** using Keras Serving or convert to GGUF for llama.cpp

---

## Resources

- **🔗 LLM-DATA-Generator**: [ElHadheqMind/LLM-DATA-Generator](https://github.com/ElHadheqMind/LLM-DATA-Generator)
- **🔗 Gemma Models on Kaggle**: [kaggle.com/models/keras/gemma3](https://www.kaggle.com/models/keras/gemma3)
- **🔗 KerasHub Documentation**: [keras.io/keras_hub](https://keras.io/keras_hub/)
- **🔗 LoRA Paper**: [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- **🔗 Keras 3 Multi-Backend**: [keras.io/keras_3](https://keras.io/keras_3/)

---

*If you found this guide helpful, consider sharing it with others who might benefit from learning about accessible LLM fine-tuning!*
