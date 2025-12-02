# 🚀 Fine-Tuning Gemma 3 with LoRA

<div align="center">

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DUnWNxLMxL3WprUqhyLayPxVr5KffTsC?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0-red.svg)](https://keras.io/)

**A comprehensive guide to fine-tuning Google's Gemma 3 models using LoRA (Low-Rank Adaptation) - works on both CPU and GPU!**

</div>

---

## 📖 About This Project

This tutorial demonstrates how to fine-tune **Gemma 3 270M** (or higher) on a custom dataset using **LoRA** - a parameter-efficient fine-tuning technique that makes training accessible on consumer hardware.

### ✨ Key Features

- 🖥️ **CPU-Friendly**: Fine-tune models without expensive GPUs
- ⚡ **GPU-Compatible**: Same code works with CUDA for faster training
- 📊 **Complete Pipeline**: From data creation to model deployment
- 🎯 **Domain Adaptation**: Example with radiobiology Q&A dataset
- 🔧 **Production-Ready**: Save and deploy your fine-tuned models

---

## 🏗️ Architecture

<div align="center">

![LoRA Architecture](lora-architecture.png)

</div>

The LoRA approach freezes the pre-trained model weights and injects trainable low-rank decomposition matrices, dramatically reducing the number of trainable parameters.

---

## 🚀 Quick Start

### Option 1: Kaggle (Recommended)
Run the notebook directly on Kaggle with free GPU/TPU access!

### Option 2: Google Colab
Click the "Open in Colab" badge above to run the notebook with free GPU access!

### Option 3: Local Installation

```bash
# Clone the repository
git clone https://github.com/ElHadheqMind/Fine-Tuning-Gemma-3.git
cd Fine-Tuning-Gemma-3

# Install dependencies
pip install keras-hub keras pandas numpy

# For GPU support (NVIDIA)
pip install keras-hub[jax] jax[cuda12]
```

---

## 📚 What You'll Learn

1. **📊 Data Creation** - Generate training data using LLM-DATA-Generator
2. **🔧 Model Setup** - Load and configure Gemma 3 with KerasHub
3. **🎛️ LoRA Configuration** - Apply parameter-efficient fine-tuning
4. **📈 Training** - Optimize hyperparameters and monitor progress
5. **🧪 Evaluation** - Compare before/after performance
6. **💾 Deployment** - Save and use your fine-tuned model

---

## 💻 Hardware Requirements

| Hardware | Gemma 3 270M | Gemma 3 1B | Gemma 3 4B |
|----------|--------------|------------|------------|
| **CPU** | ✅ ~2-4 hours | ⚠️ Slow | ❌ Not recommended |
| **GPU 8GB** | ✅ ~10 min | ✅ ~30 min | ⚠️ Tight fit |
| **GPU 16GB+** | ✅ ~5 min | ✅ ~15 min | ✅ ~45 min |

---

## 📁 Project Structure

```
Fine-Tuning-Gemma-3/
├── Fine_Tuning_Gemma3_LoRA.ipynb  # Main tutorial notebook
├── data.csv                        # Sample training data
├── lora-architecture.png           # Architecture diagram
└── README.md                       # This file
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- 🐛 Report bugs
- 💡 Suggest features
- 🔀 Submit pull requests

---

## 🙏 Acknowledgments

- [Google DeepMind](https://deepmind.google/) for the Gemma models
- [Keras Team](https://keras.io/) for KerasHub
- [Kaggle](https://www.kaggle.com/) for notebook hosting and free compute

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

Made with ❤️ by [ElHadheqMind](https://github.com/ElHadheqMind)

</div>

