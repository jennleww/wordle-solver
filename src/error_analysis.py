import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from entropy_solver import EntropySolver
from baseline_solver import BaselineSolver
from feedback import get_feedback
from tqdm import tqdm
import os
import numpy as np

def load_words(path):
    with open(path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]

def analyze_word_properties(word):
    return {
        'length': len(word),
        'unique_letters': len(set(word)),
        'vowel_count': sum(1 for c in word if c in 'aeiou'),
        'common_letters': sum(1 for c in word if c in 'etaoinshrdlu'),
        'repeated_letters': len(word) - len(set(word))
    }

def run_error_analysis(solver_class, name, solution_words, guess_words):
    solver = solver_class(solution_words, guess_words)
    results = []
    
    print(f"\nRunning {name}...")
    for answer in tqdm(solution_words, desc="Analyzing words"):
        candidates = solution_words[:]
        guesses = []
        feedbacks = []
        candidate_counts = [len(candidates)]
        
        while len(guesses) < 6:
            guess = solver.next_guess(candidates)
            guesses.append(guess)
            fb = get_feedback(guess, answer)
            feedbacks.append(fb)
            
            if fb == "GGGGG":
                break
                
            candidates = solver.prune_candidates(guess, fb, candidates)
            candidate_counts.append(len(candidates))
        
        word_props = analyze_word_properties(answer)
        
        results.append({
            'word': answer,
            'num_guesses': len(guesses),
            'success': fb == "GGGGG",
            'guesses': guesses,
            'feedbacks': feedbacks,
            'candidate_counts': candidate_counts,
            'final_candidates': len(candidates),
            **word_props
        })
    
    return pd.DataFrame(results)

def plot_candidate_reduction(df, title):
    plt.figure(figsize=(12, 6))
    
    avg_candidates = []
    for i in range(6):
        counts = [counts[i] if len(counts) > i else None for counts in df['candidate_counts']]
        valid_counts = [c for c in counts if c is not None]
        if valid_counts:
            avg_candidates.append(np.mean(valid_counts))
    
    plt.plot(range(1, len(avg_candidates) + 1), avg_candidates, marker='o')
    plt.title(f'Average Candidate Reduction - {title}')
    plt.xlabel('Guess Number')
    plt.ylabel('Average Number of Candidates')
    plt.yscale('log')
    plt.savefig(f'plots/candidate_reduction_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_word_property_analysis(df, title):
    properties = ['unique_letters', 'vowel_count', 'common_letters', 'repeated_letters']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, prop in enumerate(properties):
        sns.boxplot(data=df, x='success', y=prop, ax=axes[i])
        axes[i].set_title(f'{prop.replace("_", " ").title()} vs Success')
    
    plt.tight_layout()
    plt.savefig(f'plots/word_properties_{title.lower().replace(" ", "_")}.png')
    plt.close()

def print_error_statistics(df, name):
    print(f"\n{name} Error Analysis:")
    print(f"Total words: {len(df)}")
    print(f"Success rate: {(df['success'].sum() / len(df)) * 100:.1f}%")
    print(f"Average guesses (successful): {df[df['success']]['num_guesses'].mean():.2f}")
    print(f"Average guesses (failed): {df[~df['success']]['num_guesses'].mean():.2f}")
    
    failures = df[~df['success']]
    print("\nWord Properties in Failed Attempts:")
    for prop in ['unique_letters', 'vowel_count', 'common_letters', 'repeated_letters']:
        print(f"Average {prop}: {failures[prop].mean():.2f}")
    
    print("\nCandidate Reduction Analysis:")
    for i in range(6):
        counts = [counts[i] if len(counts) > i else None for counts in df['candidate_counts']]
        valid_counts = [c for c in counts if c is not None]
        if valid_counts:
            print(f"Guess {i+1}: {np.mean(valid_counts):.1f} candidates")

def main():
    os.makedirs('plots', exist_ok=True)
    
    solutions = load_words("data/solutions.txt")
    guesses = load_words("data/guesses.txt")
    
    entropy_df = run_error_analysis(EntropySolver, "Entropy Solver", solutions, guesses)
    baseline_df = run_error_analysis(BaselineSolver, "Baseline Solver", solutions, guesses)
    
    for df, name in [(entropy_df, "Entropy Solver"), (baseline_df, "Baseline Solver")]:
        print(f"\nGenerating error analysis for {name}...")
        
        plot_candidate_reduction(df, name)
        plot_word_property_analysis(df, name)
        
        print_error_statistics(df, name)

if __name__ == "__main__":
    main() 