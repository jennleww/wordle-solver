import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from entropy_solver import EntropySolver
from baseline_solver import BaselineSolver
from feedback import get_feedback
from tqdm import tqdm
import time

def load_words(path):
    with open(path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]

def run_solver_with_stats(solver_class, name, solution_words, guess_words):
    solver = solver_class(solution_words, guess_words)
    results = []
    
    print(f"\nRunning {name}...")
    for answer in tqdm(solution_words, desc="Evaluating words"):
        candidates = solution_words[:]
        guesses = []
        feedbacks = []
        heuristic_scores = []
        
        while len(guesses) < 6:
            guess = solver.next_guess(candidates)
            guesses.append(guess)
            fb = get_feedback(guess, answer)
            feedbacks.append(fb)
            
            if hasattr(solver, 'heuristic_score'):
                heuristic_scores.append(solver.heuristic_score(guess))
            
            if fb == "GGGGG":
                break
            candidates = solver.prune_candidates(guess, fb, candidates)
            
        results.append({
            'answer': answer,
            'num_guesses': len(guesses),
            'guesses': guesses,
            'feedbacks': feedbacks,
            'heuristic_scores': heuristic_scores,
            'success': fb == "GGGGG"
        })
    
    return pd.DataFrame(results)

def plot_guess_distribution(df, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='num_guesses', bins=range(1, 8))
    plt.title(f'Distribution of Number of Guesses - {title}')
    plt.xlabel('Number of Guesses')
    plt.ylabel('Count')
    plt.savefig(f'plots/guess_distribution_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_letter_frequency(df, title):
    all_guesses = ''.join([''.join(guesses) for guesses in df['guesses']])
    letter_counts = Counter(all_guesses)
    
    plt.figure(figsize=(12, 6))
    letters = sorted(letter_counts.keys())
    counts = [letter_counts[l] for l in letters]
    
    plt.bar(letters, counts)
    plt.title(f'Letter Frequency in Guesses - {title}')
    plt.xlabel('Letter')
    plt.ylabel('Frequency')
    plt.savefig(f'plots/letter_frequency_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_position_heatmap(df, title):
    position_matrix = np.zeros((5, 26))
    
    for guesses in df['guesses']:
        for guess in guesses:
            for pos, letter in enumerate(guess):
                position_matrix[pos][ord(letter) - ord('a')] += 1
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(position_matrix, 
                xticklabels=[chr(i) for i in range(ord('a'), ord('z')+1)],
                yticklabels=['Position 1', 'Position 2', 'Position 3', 'Position 4', 'Position 5'],
                cmap='YlOrRd')
    plt.title(f'Letter Position Heatmap - {title}')
    plt.savefig(f'plots/position_heatmap_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_heuristic_vs_performance(df, title):
    if 'heuristic_scores' not in df.columns:
        return
        
    plt.figure(figsize=(10, 6))
    
    avg_scores = []
    for i in range(6):
        scores = [scores[i] if len(scores) > i else None for scores in df['heuristic_scores']]
        valid_scores = [s for s in scores if s is not None]
        if valid_scores:
            avg_scores.append(np.mean(valid_scores))
    
    plt.plot(range(1, len(avg_scores) + 1), avg_scores, marker='o')
    plt.title(f'Average Heuristic Score by Guess Number - {title}')
    plt.xlabel('Guess Number')
    plt.ylabel('Average Heuristic Score')
    plt.savefig(f'plots/heuristic_scores_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_success_by_guess(df, title):
    plt.figure(figsize=(10, 6))
    
    success_by_guess = []
    for i in range(1, 7):
        success_rate = (df['num_guesses'] == i).mean() * 100
        success_by_guess.append(success_rate)
    
    plt.bar(range(1, 7), success_by_guess)
    plt.title(f'Success Rate by Guess Number - {title}')
    plt.xlabel('Guess Number')
    plt.ylabel('Success Rate (%)')
    plt.savefig(f'plots/success_by_guess_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_comparison(entropy_df, baseline_df):
    plt.figure(figsize=(10, 6))
    
    # Calculate success rates for each guess number
    entropy_rates = []
    baseline_rates = []
    
    for i in range(1, 7):
        entropy_rate = (entropy_df['num_guesses'] == i).mean() * 100
        baseline_rate = (baseline_df['num_guesses'] == i).mean() * 100
        entropy_rates.append(entropy_rate)
        baseline_rates.append(baseline_rate)
    
    # Create the comparison plot
    x = np.arange(6)  # the label locations
    width = 0.35  # the width of the bars
    
    plt.bar(x - width/2, entropy_rates, width, label='Entropy Solver')
    plt.bar(x + width/2, baseline_rates, width, label='Baseline Solver')
    
    plt.title('Success Rate Comparison by Guess Number')
    plt.xlabel('Guess Number')
    plt.ylabel('Success Rate (%)')
    plt.xticks(x, range(1, 7))
    plt.legend()
    
    plt.savefig('plots/solver_comparison.png')
    plt.close()

def main():
    import os
    os.makedirs('plots', exist_ok=True)
    
    solutions = load_words("data/solutions.txt")
    guesses = load_words("data/guesses.txt")
    
    entropy_df = run_solver_with_stats(EntropySolver, "Entropy Solver", solutions, guesses)
    baseline_df = run_solver_with_stats(BaselineSolver, "Baseline Solver", solutions, guesses)

    for df, name in [(entropy_df, "Entropy Solver"), (baseline_df, "Baseline Solver")]:
        print(f"\nGenerating visualizations for {name}...")
        
        plot_guess_distribution(df, name)
        plot_letter_frequency(df, name)
        plot_position_heatmap(df, name)
        plot_success_by_guess(df, name)
        plot_heuristic_vs_performance(df, name)
        
        print(f"\n{name} Statistics:")
        print(f"Average guesses: {df['num_guesses'].mean():.2f}")
        print(f"Success rate: {(df['success'].sum() / len(df)) * 100:.1f}%")
        print(f"Most common number of guesses: {df['num_guesses'].mode()[0]}")
        print(f"Standard deviation: {df['num_guesses'].std():.2f}")
    
    plot_comparison(entropy_df, baseline_df)

if __name__ == "__main__":
    main() 