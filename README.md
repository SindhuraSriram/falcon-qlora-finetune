# 🦅 Fine-Tuning Falcon Model on Dolly Dataset with QLoRA

This project demonstrates how to fine-tune the **Falcon-RW-1B** language model on the **Dolly 15K dataset** using **QLoRA (Quantized Low-Rank Adapter)** for efficient training on limited hardware like Google Colab (T4 GPU).

---

## 📌 Project Overview

We leverage the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework to run lightweight and memory-efficient fine-tuning of Falcon models using QLoRA — ideal for training even large models on consumer hardware or free GPU services.

---

## 🚀 What You’ll Learn

- How to set up a training environment with LLaMA-Factory
- How QLoRA enables efficient training on a low-memory GPU
- How to fine-tune Falcon-RW-1B on the instruction-following Dolly dataset
- How to save, test, and evaluate the fine-tuned model

---

## 🛠️ Environment Setup

```bash
# Clone LLaMA-Factory and install dependencies
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -r requirements.txt
```

---

## 📂 Dataset

We use the **Dolly 15K** dataset from Databricks, a high-quality instruction-following dataset for training helpful assistants.

> If you're using Colab, LLaMA-Factory auto-downloads it when `dataset='dolly'` is specified.

---

## ⚙️ Model & Training Configuration

```python
# Sample configuration used (in YAML or CLI-style)
model_name_or_path = "tiiuae/falcon-rw-1b"
adapter_type = "lora"
quantization_bit = 4
dataset = "dolly"
output_dir = "./falcon-dolly-qlora"
num_train_epochs = 3
per_device_train_batch_size = 2
```

Training is done using QLoRA, which applies:
- 4-bit quantization to the base model
- LoRA adapters for efficient tuning

---

## 🧪 Inference Example

Once trained, you can test your model like this:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./falcon-dolly-qlora")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")

prompt = "Explain how photosynthesis works."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🧠 Why QLoRA?

QLoRA enables:
- 64–80% memory reduction via 4-bit quantization
- Training 1B+ parameter models on GPUs with <16GB VRAM
- Faster and cheaper experimentation

Perfect for Colab and academic/research workflows!

---

## 📊 Training Summary

- **Model**: Falcon-RW-1B
- **Dataset**: Dolly 15K (Instruction tuning)
- **Fine-tuning method**: QLoRA (4-bit quantization + LoRA adapters)
- **Platform**: Google Colab (T4 GPU)
- **Epochs**: 3
- **Training Time**: ~2.3 hours
- **VRAM used**: ~11.5 GB
- **Final model size**: ~1.6 GB (with LoRA)

---

## 💡 Observations

- QLoRA made it possible to train Falcon 1B on a free Colab T4.
- No OOM errors throughout training.
- Dolly dataset worked out-of-the-box with LLaMA-Factory.

---

## 💾 Output

- Fine-tuned model saved at: `./falcon-dolly-qlora`
- Also available on HuggingFace Hub:  
  👉 [SindhuraSriram/falcon-rw-1b-dolly-qlora](https://huggingface.co/SindhuraSriram/falcon-rw-1b-dolly-qlora)

---

## 📎 References

- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [Falcon-RW-1B on HuggingFace](https://huggingface.co/tiiuae/falcon-rw-1b)
- [Dolly Dataset](https://huggingface.co/datasets/databricks/dolly)

---

## 🙌 Acknowledgements

Thanks to HuggingFace, Databricks, and the LLaMA-Factory contributors for enabling open, accessible model training for all.

---

## 📬 Contact

For any questions, feel free to reach out:

**Sindhura Sriram**  
📧 sindhura.sriram@ufl.edu  
🔗 [LinkedIn](https://www.linkedin.com/in/sindhura-sriram) | 🌐 [Portfolio](https://www.sindhura-sriram.com)
