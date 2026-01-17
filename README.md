# Qwen2-VL Fine-Tuning for LaTeX OCR aj

This repository contains a Jupyter Notebook (`finetuning_Qwen_VL.ipynb`) demonstrating how to efficiently fine-tune the **Qwen2-VL-7B-Instruct** Vision-Language Model (VLM) using **Unsloth**.

The specific use case demonstrated is **Optical Character Recognition (OCR) for Mathematics**, converting images of equations into their corresponding **LaTeX** representation.

##  Features

* **Efficient Training**: Uses [Unsloth](https://github.com/unslothai/unsloth) for 2x faster training and 60% less memory usage.
* **4-bit Quantization**: Loads the model in 4-bit precision to fit on consumer GPUs (e.g., Google Colab T4).
* **Full VLM Fine-Tuning**: Targets both vision and language layers via LoRA (Low-Rank Adaptation).
* **Dataset**: Utilizes the `unsloth/Latex_OCR` dataset.

##  Installation

To run this notebook, you will need to install the following dependencies. The notebook runs best in a Linux environment with NVIDIA GPUs (e.g., Google Colab).

```bash
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
pip install --no-deps unsloth
