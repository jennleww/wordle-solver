import math
from feedback import get_feedback, matches_feedback
from collections import Counter
from functools import lru_cache

class EntropySolver:
    def __init__(self, solution_words, allowed_words):
        self.solutions = solution_words
        self.allowed = allowed_words
        self.letter_freqs = Counter(''.join(solution_words))
        self.position_freqs = [Counter(word[i] for word in solution_words) for i in range(5)]
        self.vowels = set('aeiou')
        
        self.common_letters = set(letter for letter, count in self.letter_freqs.most_common(10))
        self.common_positions = [set(pos.most_common(5)[i][0] for i in range(5)) for pos in self.position_freqs]
        
        self.scored_words = [(word, self.heuristic_score(word)) for word in self.allowed]
        self.top_words = [word for word, _ in sorted(self.scored_words, key=lambda x: x[1], reverse=True)[:20]]
        
        self._feedback_cache = {}
        self._entropy_cache = {}

    def prune_candidates(self, guess, feedback, candidates):
        return [word for word in candidates if matches_feedback(word, guess, feedback)]

    def heuristic_score(self, word):
        score = 0
        for letter in word:
            score += self.letter_freqs[letter]
        
        for i, letter in enumerate(word):
            score += self.position_freqs[i][letter] * 2
        
        letter_counts = Counter(word)
        for letter, count in letter_counts.items():
            if count > 1 and self.letter_freqs[letter] < 100:
                score -= 50 * (count - 1)
        
        vowel_count = sum(1 for c in word if c in self.vowels)
        if 1 <= vowel_count <= 3:
            score += 100
            
        for i, letter in enumerate(word):
            if letter in self.common_positions[i]:
                score += 50
        
        return score

    @lru_cache(maxsize=1024)
    def get_cached_feedback(self, guess, solution):
        return get_feedback(guess, solution)

    def score_entropy(self, guess, candidates):
        cache_key = (guess, len(candidates))
        if cache_key in self._entropy_cache:
            return self._entropy_cache[cache_key]
            
        feedback_counts = {}
        for solution in candidates:
            fb = self.get_cached_feedback(guess, solution)
            feedback_counts[fb] = feedback_counts.get(fb, 0) + 1
            
        total = sum(feedback_counts.values())
        entropy = -sum((n/total) * math.log2(n/total) for n in feedback_counts.values())
        
        self._entropy_cache[cache_key] = entropy
        return entropy

    def next_guess(self, candidates):
        if len(candidates) == len(self.solutions):
            return self.top_words[0]
            
        possible_guesses = [word for word in candidates if word in self.allowed]
        if not possible_guesses:
            possible_guesses = self.allowed
            
        scored_guesses = [(word, self.heuristic_score(word)) for word in possible_guesses]
        top_guesses = [word for word, _ in sorted(scored_guesses, key=lambda x: x[1], reverse=True)[:10]]
        
        best_guess, best_score = None, -1
        for word in top_guesses:
            score = self.score_entropy(word, candidates)
            if score > best_score:
                best_guess, best_score = word, score
                
        return best_guess
