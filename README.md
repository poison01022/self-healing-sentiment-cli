# 💬 Self-Healing Sentiment Classifier (CLI App)

A robust, modular CLI-based sentiment classification system that automatically handles low-confidence predictions using fallback logic — powered by a fine-tuned BERT model, LangGraph, and a backup zero-shot classifier.

---

## 📌 Features

- 🔍 **Sentiment Analysis** (Positive/Negative) using a fine-tuned `DistilBERT` model on the IMDb dataset.
- 🧠 **Self-Healing Logic**: If model confidence is low, fallback options kick in:
  - ✅ Backup zero-shot model (`facebook/bart-large-mnli`)
  - 👤 Manual user clarification (optional version)
- 📈 **Confidence Visualization**: See how confident the model is across inputs.
- 📊 **Fallback Statistics**: Live histogram showing how often fallback was needed.
- 💾 **Logging**: All interactions (input, prediction, confidence, correction) are saved.

---

## 🚀 How It Works

The application is built as a **DAG (Directed Acyclic Graph)** using `LangGraph`, with the following pipeline:

User Input ➡️ Inference ➡️ Confidence Check
⬇️
[Low Confidence Detected]
⬇️
Backup Model Prediction
⬇️
Final Output & Logging


---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/poison01022/self-healing-sentiment-cli.git
cd self-healing-sentiment-cli
```
