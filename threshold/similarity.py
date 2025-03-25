from difflib import SequenceMatcher
from itertools import combinations

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def compute_avg_similarity(choices):
    if len(choices) < 2:
        return 0  # No similarity can be computed for a single choice
    total_sim = sum(similar(a, b) for a, b in combinations(choices, 2))
    avg_sim = total_sim / len(list(combinations(choices, 2)))
    return avg_sim