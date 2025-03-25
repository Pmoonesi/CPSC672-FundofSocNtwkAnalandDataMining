import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from similarity import compute_avg_similarity


def parse_input(lines):
    results = {}
    current_id = None
    choices = []
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("id is "):
            if current_id is not None:
                avg_sim = compute_avg_similarity(choices)
                results[current_id] = (choices, selected_index, avg_sim)
            
            current_id = line.split("id is ")[1].strip(":")
            choices = []
            selected_index = None
        elif line.startswith(tuple(str(i) + ")" for i in range(10))):
            choices.append(line.split(") ", 1)[1])
        elif line.startswith("Input: "):
            selected_index = int(line.split("Input: ")[1])
    
    if current_id is not None:
        avg_sim = compute_avg_similarity(choices)
        results[current_id] = (choices, selected_index, avg_sim)
    
    return results

def analyze_results(results):
    none_selected = []
    other_selected = []
    avg_similarities = {}
    # labels = {}
    
    for key, (choices, selected_index, avg_sim) in results.items():
        try:
            avg_similarities[key] = avg_sim
            if choices[selected_index] == "None.":
                none_selected.append(key)
                # labels[key] = 0
            else:
                other_selected.append(key)
                # labels[key] = 1
        except Exception as e:
            pass
    
    return none_selected, other_selected, avg_similarities
    # return labels, avg_similarities

def draw_roc(none_selected, other_selected, avg_similarities):
    # Prepare data for ROC analysis
    similarity_scores = []
    labels = []

    for key, avg_sim in avg_similarities.items():
        label = 0 if key in none_selected else 1  # 0 if "None" was chosen, 1 otherwise
        similarity_scores.append(avg_sim)
        labels.append(label)

    # Convert to numpy arrays
    similarity_scores = np.array(similarity_scores)
    labels = np.array(labels)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, similarity_scores)
    roc_auc = auc(fpr, tpr)

    # Find the best threshold using Youden's J statistic
    j_scores = tpr - fpr
    best_threshold = thresholds[np.argmax(j_scores)]

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.scatter(fpr[np.argmax(j_scores)], tpr[np.argmax(j_scores)], marker="o", color="red", label=f"Best Threshold = {best_threshold:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()

    print(f"Optimal similarity threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    input_data = sys.stdin.read().splitlines()
    results = parse_input(input_data)
    none_selected, other_selected, avg_similarities = analyze_results(results)
    # labels, avg_similarities = analyze_results(results)
    
    # print("Groups where 'None' was selected:", none_selected)
    # print("Groups where another option was selected:", other_selected)
    # print("Average similarity per group:", avg_similarities)

    draw_roc(none_selected, other_selected, avg_similarities)
