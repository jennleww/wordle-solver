
def get_feedback(guess, solution):
    feedback = ['.'] * 5
    taken = [False] * 5
    for i in range(5):
        if guess[i] == solution[i]:
            feedback[i] = 'G'
            taken[i] = True
    for i in range(5):
        if feedback[i] == 'G':
            continue
        for j in range(5):
            if not taken[j] and guess[i] == solution[j]:
                feedback[i] = 'Y'
                taken[j] = True
                break
        if feedback[i] == '.':
            feedback[i] = 'B'
    return ''.join(feedback)

def matches_feedback(word, guess, feedback):
    return get_feedback(guess, word) == feedback
