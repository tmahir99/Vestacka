from abc import ABC, abstractmethod
import math
from typing import List, Tuple

class Agent(ABC):
    @abstractmethod
    def get_chosen_column(self, state: List[List[int]], max_depth: int) -> int:
        pass

    def evaluate_state(self, state: List[List[int]], player: int) -> int:
        score = 0
        opponent = 3 - player
        difficulty_multiplier = {
            'easy': 1.0,
            'medium': 1.5,
            'hard': 2.0
        }[self.difficulty]

        positional_weights = [
            [1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],  # Bottom row
            [1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2],
            [1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1],
            [1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1],
            [1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0],
            [1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0]   # Top row
        ]

        row_weights = [1.5, 1.4, 1.3, 1.3, 1.3, 1.3] 

        # Horizontal windows
        for row in range(6):
            for col in range(4):
                window = state[row][col:col+4]
                score += (
                    self._evaluate_window(window, player, opponent)
                    * positional_weights[row][col]
                    * row_weights[row]
                    * difficulty_multiplier
                )

        for row in range(3):
            for col in range(7):
                window = [state[row+i][col] for i in range(4)]
                score += (
                    self._evaluate_window(window, player, opponent)
                    * positional_weights[row][col]
                    * row_weights[row]
                    * 1.5 
                    * difficulty_multiplier
                )

        for row in range(3):
            for col in range(4):
                pos_diagonal = [state[row+i][col+i] for i in range(4)]
                neg_diagonal = [state[row+3-i][col+i] for i in range(4)]
                score += (
                    self._evaluate_window(pos_diagonal, player, opponent)
                    * positional_weights[row][col]
                    * row_weights[row]
                    * 1.2  
                    * difficulty_multiplier
                )
                score += (
                    self._evaluate_window(neg_diagonal, player, opponent)
                    * positional_weights[row][col]
                    * row_weights[row]
                    * 1.2
                    * difficulty_multiplier
                )

        return score

    def _evaluate_window(self, window: List[int], player: int, opponent: int) -> int:
        score = 0
        player_count = window.count(player)
        empty_count = window.count(0)
        opponent_count = window.count(opponent)

        if player_count == 4:
            score += 1000
        elif player_count == 3 and empty_count == 1:
            score += 50
        elif player_count == 2 and empty_count == 2:
            score += 20
        elif player_count == 1 and empty_count == 3:
            score += 5

        if opponent_count == 3 and empty_count == 1:
            score -= 300
        elif opponent_count == 2 and empty_count == 2:
            score -= 90
        elif opponent_count == 1 and empty_count == 3:
            score -= 80

        return score

class MinimaxABAgent(Agent):
    def _minimax(self, state: List[List[int]], depth: int, maximizing: bool, 
                 alpha: float, beta: float) -> float:
        valid_moves = [col for col in range(7) if any(state[row][col] == 0 for row in range(6))]

        if not valid_moves:
            raise ValueError("No valid moves left!")
        
        if depth == 0 or not valid_moves or self._is_terminal_node(state):
            return self.evaluate_state(state, 1 if maximizing else 2)

        if maximizing:
            value = -math.inf
            for move in valid_moves:
                new_state = self._make_move(state, move, 1)
                value = max(value, self._minimax(new_state, depth-1, False, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for move in valid_moves:
                new_state = self._make_move(state, move, 2)
                value = min(value, self._minimax(new_state, depth-1, True, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def _make_move(self, state: List[List[int]], col: int, player: int) -> List[List[int]]:
        if all(state[row][col] != 0 for row in range(6)):
            raise ValueError(f"Cannot make move in column {col} - column is full")
        
        new_state = [row[:] for row in state]
        
        for row in range(5, -1, -1):
            if new_state[row][col] == 0:
                new_state[row][col] = player
                return new_state
        
        raise ValueError(f"Unexpected error in column {col}")

    def _is_terminal_node(self, state: List[List[int]]) -> bool:
        for row in range(6):
            for col in range(4):
                if (state[row][col] != 0 and
                    state[row][col] == state[row][col+1] == 
                    state[row][col+2] == state[row][col+3]):
                    return True

        for row in range(3):
            for col in range(7):
                if (state[row][col] != 0 and
                    state[row][col] == state[row+1][col] == 
                    state[row+2][col] == state[row+3][col]):
                    return True

        for row in range(3):
            for col in range(4):
                if (state[row][col] != 0 and
                    state[row][col] == state[row+1][col+1] == 
                    state[row+2][col+2] == state[row+3][col+3]):
                    return True

        for row in range(3, 6):
            for col in range(4):
                if (state[row][col] != 0 and
                    state[row][col] == state[row-1][col+1] == 
                    state[row-2][col+2] == state[row-3][col+3]):
                    return True

        return False

class DifficultyAgent(Agent):
    def __init__(self, difficulty: str):
        self.difficulty = difficulty.lower()

        if self.difficulty == "easy":
            self.max_depth = 4
        elif self.difficulty == "medium":
            self.max_depth = 6
        elif self.difficulty == "hard":
            self.max_depth = 7
        else:
            raise ValueError("Invalid difficulty level. Choose 'easy', 'medium', or 'hard'.")

    def get_chosen_column(self, state: List[List[int]], max_depth: int = None) -> int:
        if max_depth is None:
            max_depth = self.max_depth

        valid_moves =valid_moves = [col for col in range(7) if state[0][col] == 0]
        
        if not valid_moves:
            raise ValueError("No valid moves left!")

        column_weights = {
            'easy': {3: 2.0, 2: 2.0, 4: 2.0, 1: 2.0, 5: 2.0, 0: 2.0, 6: 2.0},
            'medium': {3: 2.5, 2: 2.5, 4: 2.5, 1: 2.5, 5: 2.5, 0: 2.5, 6: 2.5},
            'hard': {3: 3.0, 2: 3.0, 4: 3.0, 1: 3.0, 5: 3.0, 0: 3.0, 6: 3.0}
        }[self.difficulty]

        best_score = -math.inf
        best_move = valid_moves[0]
        alpha = -math.inf
        beta = math.inf

        for move in valid_moves:
            new_state = self._make_move(state, move, 1)
            base_score = self._minimax(new_state, max_depth - 1, False, alpha, beta)
            weighted_score = base_score * column_weights[move]

            if weighted_score > best_score:
                best_score = weighted_score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move

    _minimax = MinimaxABAgent._minimax
    _make_move = MinimaxABAgent._make_move
    _is_terminal_node = MinimaxABAgent._is_terminal_node
