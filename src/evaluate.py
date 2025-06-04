from entropy_solver import EntropySolver
from baseline_solver import BaselineSolver
from feedback import get_feedback
from tqdm import tqdm
import time

def load_words(path):
    with open(path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]

def run_solver(solver_class, name, solution_words, guess_words):
    solver = solver_class(solution_words, guess_words)
    total_guesses, failures = 0, 0
    start_time = time.time()
    
    print(f"\nRunning {name}...")
    for answer in tqdm(solution_words, desc="Evaluating words"):
        candidates = solution_words[:]
        guesses = 0
        while guesses < 6:
            guess = solver.next_guess(candidates)
            guesses += 1
            fb = get_feedback(guess, answer)
            if fb == "GGGGG":
                break
            candidates = solver.prune_candidates(guess, fb, candidates)
        else:
            failures += 1
        total_guesses += guesses
    
    elapsed_time = time.time() - start_time
    avg_guesses = total_guesses/len(solution_words)
    failure_rate = failures/len(solution_words)*100
    
    print(f"\n{name} Results:")
    print(f"Average guesses: {avg_guesses:.2f}")
    print(f"Failure rate: {failure_rate:.1f}%")
    print(f"Time taken: {elapsed_time:.1f} seconds")
    print(f"Average time per word: {elapsed_time/len(solution_words):.3f} seconds")

if __name__ == "__main__":
    solutions = load_words("data/solutions.txt")
    guesses = load_words("data/guesses.txt")
    print(f"Loaded {len(solutions)} solution words and {len(guesses)} guess words")
    run_solver(EntropySolver, "Entropy Solver", solutions, guesses)
    run_solver(BaselineSolver, "Baseline Solver", solutions, guesses)
