from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
from langgraph.graph import StateGraph
from typing import TypedDict, Literal, Optional

# Define the state schema
class State(TypedDict):
    text: str
    prediction: Optional[str]
    confidence: Optional[float]
    status: Optional[Literal["accepted", "fallback", "corrected", "backup_used"]]
    corrected: Optional[str]

# Load fine-tuned model
model_path = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load backup zero-shot model
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

label_map = {0: "Negative", 1: "Positive"}

# Inference node
def inference_node(state: State) -> State:
    input_text = state["text"]
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        confidence, pred_label = torch.max(probs, dim=1)

    pred_label_text = label_map[pred_label.item()]
    print(f"[InferenceNode] Predicted label: {pred_label_text} | Confidence: {confidence.item() * 100:.0f}%")

    state["prediction"] = pred_label_text
    state["confidence"] = confidence.item()
    return state

# Confidence check
def confidence_check_node(state: State) -> State:
    threshold = 0.75
    if state["confidence"] >= threshold:
        state["status"] = "accepted"
    else:
        print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
        state["status"] = "fallback"
    return state

# Fallback using zero-shot
def fallback_node(state: State) -> State:
    print(f"[FallbackNode] Attempting backup model on uncertain prediction...")

    result = zero_shot(state["text"], candidate_labels=["Positive", "Negative"])
    label = result["labels"][0]
    confidence = result["scores"][0]

    print(f"[BackupModel] Zero-shot predicted: {label} (Confidence: {confidence * 100:.0f}%)")

    state["corrected"] = label
    state["status"] = "backup_used"
    return state

# DAG setup
builder = StateGraph(State)

builder.add_node("InferenceNode", inference_node)
builder.add_node("ConfidenceCheckNode", confidence_check_node)
builder.add_node("FallbackNode", fallback_node)

builder.set_entry_point("InferenceNode")
builder.add_edge("InferenceNode", "ConfidenceCheckNode")

def router(state: State):
    if state["status"] == "fallback":
        return "FallbackNode"
    return "__end__"

builder.add_conditional_edges("ConfidenceCheckNode", router)

graph = builder.compile()
