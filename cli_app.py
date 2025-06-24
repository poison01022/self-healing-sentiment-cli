from dag_langgraph import graph
import json
import os
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt

LOG_FILE = "logs/classification_log.jsonl"
os.makedirs("logs", exist_ok=True)

confidence_history = []

def log_interaction(state):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "input": state["text"],
        "prediction": state.get("prediction"),
        "confidence": state.get("confidence"),
        "status": state.get("status"),
        "corrected": state.get("corrected", None)
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    if state.get("confidence"):
        confidence_history.append(state["confidence"])

def print_fallback_stats():
    with open(LOG_FILE) as f:
        entries = [json.loads(line) for line in f]
    counts = Counter(e["status"] for e in entries)
    print("\n📊 Fallback Frequency Stats:")
    for status, count in counts.items():
        bar = "█" * count
        print(f"{status:12}: {bar} ({count})")

def plot_confidence_curve():
    if not confidence_history:
        print("No confidence data to plot.")
        return
    plt.plot(confidence_history, marker="o", label="Confidence")
    plt.axhline(0.75, color="red", linestyle="--", label="Threshold")
    plt.title("Confidence Curve Over Inputs")
    plt.xlabel("Input Index")
    plt.ylabel("Confidence Score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    print("💬 Sentiment Classifier (Self-Healing)")
    print("Type 'exit' to quit, 'stats' for fallback chart, or 'plot' for confidence curve.\n")

    while True:
        user_input = input("Enter a sentence: ")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "stats":
            print_fallback_stats()
            continue
        if user_input.lower() == "plot":
            plot_confidence_curve()
            continue

        state = {"text": user_input}
        final_state = graph.invoke(state)

        if final_state.get("status") == "corrected":
            print(f"Final Label: {final_state['corrected']} (Corrected via user clarification)")
        elif final_state.get("status") == "backup_used":
            print(f"Final Label: {final_state['corrected']} (Backup Model Used)")
        else:
            print(f"Final Label: {final_state['prediction']} (Confidence: {final_state['confidence'] * 100:.0f}%)")

        log_interaction(final_state)
        print()

if __name__ == "__main__":
    main()
