
from feedback import get_feedback, matches_feedback

class BaselineSolver:
    def __init__(self, solution_words, allowed_words):
        self.solutions = solution_words
        self.allowed = allowed_words
        self.start_word = "arise"

    def prune_candidates(self, guess, feedback, candidates):
        return [word for word in candidates if matches_feedback(word, guess, feedback)]

    def next_guess(self, candidates):
        return self.start_word if candidates == self.solutions else candidates[0]
