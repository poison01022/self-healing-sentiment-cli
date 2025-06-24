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

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv env
env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 📥 Model Download Instructions

The sentiment classifier expects a fine-tuned **DistilBERT** model saved at `./fine_tuned_model/`.

You can download it directly from the terminal using the Hugging Face Transformers library:

#### Step 1: (Optional) Authenticate with Hugging Face if using a private repo
```bash
transformers-cli login
```

### Step 2: Download and save the model locally

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "YOUR_HF_USERNAME/fine-tuned-distilbert-imdb"    #🔁 Replace YOUR_HF_USERNAME/fine-tuned-distilbert-imdb with the actual model path from Hugging Face Hub.

AutoTokenizer.from_pretrained(model_name).save_pretrained("fine_tuned_model")
AutoModelForSequenceClassification.from_pretrained(model_name).save_pretrained("fine_tuned_model")
```

### Step 3: (Optional) Pre-download the Backup Model

The fallback logic uses Hugging Face's facebook/bart-large-mnli model for zero-shot classification.

You can pre-download it to avoid delays during runtime:

```python
from transformers import pipeline

pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
```

### 🚀 Run the CLI App

Before launching the CLI, make sure the fine-tuned model is available.  
If you haven't trained it yet, run:

```bash
python fine_tune_sentiments.py
```
This will fine-tune DistilBERT on the IMDb dataset and save the model to the fine_tuned_model/ directory.

Then, start the sentiment classification CLI application:

```bash
python cli_app.py
```

The CLI supports intelligent fallback using a zero-shot model and logs all predictions with confidence scores.



🛠️ Commands during execution:

exit → Quit the app

stats → Show fallback usage statistics

plot → Display confidence curve over recent inputs

### OUTPUT SAMPLE

![Image](https://github.com/user-attachments/assets/9e3f3848-fefb-4fb6-8d0e-768008478140)
![Image](https://github.com/user-attachments/assets/051d2eca-b4db-44e1-a8c9-09adaf810021)


### 📁 Folder Structure

```arduino

├── cli_app.py
├── dag_langgraph.py
├── fine_tune_sentiments.py  # (optional, for training)
├── logs/
│   └── classification_log.jsonl
├── fine_tuned_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer/
├── requirements.txt
└── README.md
```

---

## 🧠 Credits

- Fine-tuned using the IMDb dataset  
- Backup via [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli)  
- Built with [Hugging Face Transformers](https://huggingface.co) and [LangGraph](https://python.langchain.com/docs/langgraph)  
- 💻 Developed by **Adarsh Prasad**

---
