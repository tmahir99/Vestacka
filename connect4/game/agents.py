from abc import ABC, abstractmethod
import math
from typing import List, Tuple, Dict
from enum import Enum

class HeuristicType(Enum):
    STANDARD = "standard"
    FIRST_BEST = "first_best"
    MANHATTAN = "manhattan"

class Agent(ABC):
    @abstractmethod
    def get_chosen_column(self, state: List[List[int]], max_depth: int) -> int:
        pass

    def evaluate_state(self, state: List[List[int]], player: int, heuristic_type: HeuristicType = HeuristicType.STANDARD) -> int:
        if heuristic_type == HeuristicType.FIRST_BEST:
            return self._first_best_heuristic(state, player)
        elif heuristic_type == HeuristicType.MANHATTAN:
            return self._manhattan_heuristic(state, player)
        else:
            return self._standard_heuristic(state, player)

    def _standard_heuristic(self, state: List[List[int]], player: int) -> int:
        score = 0
        opponent = 3 - player
        
        for row in range(6):
            for col in range(4):
                window = state[row][col:col+4]
                score += self._evaluate_window(window, player, opponent)

        for row in range(3):
            for col in range(7):
                window = [state[row+i][col] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)

        for row in range(3):
            for col in range(4):
                window = [state[row+i][col+i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)

        for row in range(3, 6):
            for col in range(4):
                window = [state[row-i][col+i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)

        return score

    def _first_best_heuristic(self, state: List[List[int]], player: int) -> int:
        """
        First-Best heuristic implementation that prioritizes:
        1. Immediate wins
        2. Blocking opponent wins
        3. Creating potential winning sequences
        4. Strategic position control
        """
        score = 0
        opponent = 3 - player
        weights = {
        
            'win': 100000,    
            'block': 90000,         
            'threat': 1000,         
            'position': 100,  
            'center': 50,           
            'connection': 30  
        }
        
        for col in range(7):
            if self._is_valid_move(state, col):
                test_state = self._simulate_move(state, col, player)
                if self._is_winning_state(test_state, player):
                    return weights['win']
                
                test_state = self._simulate_move(state, col, opponent)
                if self._is_winning_state(test_state, opponent):
                    score += weights['block']

        for row in range(6):
            for col in range(7):
                if state[row][col] == player:
                    position_score = (row + 1) * 10  
                    score += position_score * weights['position'] // 100
                    
                    if 2 <= col <= 4:
                        score += weights['center']
                    
                    for dx, dy in [(0,1), (1,0), (1,1), (1,-1)]:
                        connected = 0
                        for i in range(1, 4):
                            new_row, new_col = row + dy*i, col + dx*i
                            if (0 <= new_row < 6 and 0 <= new_col < 7 and 
                                state[new_row][new_col] == player):
                                connected += 1
                            else:
                                break
                        score += connected * weights['connection']

        for row in range(6):
            for col in range(4):
                window = state[row][col:col+4]
                score += self._evaluate_threat(window, player) * weights['threat']

        return score

    def _manhattan_heuristic(self, state: List[List[int]], player: int) -> int:
        """
        Manhattan distance heuristic that evaluates positions based on:
        1. Distance to center
        2. Distance to other friendly pieces
        3. Distance to winning positions
        4. Distance to blocking positions
        """
        score = 0
        opponent = 3 - player
        center_col = 3
        
        weights = {
            'center_distance': 30,
            'piece_proximity': 20,
            'winning_distance': 100,
            'blocking_distance': 80,
            'vertical_position': 40
        }
        
        for row in range(6):
            for col in range(7):
                if state[row][col] == player:
                    center_distance = abs(col - center_col)
                    score -= center_distance * weights['center_distance']
                    
                    score += (5 - row) * weights['vertical_position']
                    
                    for r in range(6):
                        for c in range(7):
                            if state[r][c] == player and (r != row or c != col):
                                distance = abs(row - r) + abs(col - c)
                                if distance <= 3: 
                                    score += (4 - distance) * weights['piece_proximity']
                    
                    winning_positions = self._find_winning_positions(state, player)
                    for win_row, win_col in winning_positions:
                        distance = abs(row - win_row) + abs(col - win_col)
                        score += (7 - distance) * weights['winning_distance']
                    
                    blocking_positions = self._find_winning_positions(state, opponent)
                    for block_row, block_col in blocking_positions:
                        distance = abs(row - block_row) + abs(col - block_col)
                        score += (7 - distance) * weights['blocking_distance']

        return score

    def _find_winning_positions(self, state: List[List[int]], player: int) -> List[Tuple[int, int]]:
        """Find positions that could lead to a win"""
        winning_positions = []
        
        for col in range(7):
            for row in range(5, -1, -1):
                if state[row][col] == 0:
                    test_state = [r[:] for r in state]
                    test_state[row][col] = player
                    
                    if self._is_winning_state(test_state, player):
                        winning_positions.append((row, col))
                    break
                    
        return winning_positions

    def _simulate_move(self, state: List[List[int]], col: int, player: int) -> List[List[int]]:
        """Simulate making a move in a column"""
        new_state = [row[:] for row in state]
        for row in range(5, -1, -1):
            if new_state[row][col] == 0:
                new_state[row][col] = player
                break
        return new_state

    def _is_valid_move(self, state: List[List[int]], col: int) -> bool:
        """Check if a move is valid"""
        return state[0][col] == 0

    def _is_winning_state(self, state: List[List[int]], player: int) -> bool:
        """Check if the state is a win for the player"""
        for row in range(6):
            for col in range(4):
                if all(state[row][col+i] == player for i in range(4)):
                    return True

        for row in range(3):
            for col in range(7):
                if all(state[row+i][col] == player for i in range(4)):
                    return True

        for row in range(3):
            for col in range(4):
                if all(state[row+i][col+i] == player for i in range(4)):
                    return True

        for row in range(3, 6):
            for col in range(4):
                if all(state[row-i][col+i] == player for i in range(4)):
                    return True

        return False

    def _evaluate_window(self, window: List[int], player: int, opponent: int) -> int:
        """Evaluate a window of four positions"""
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

    def _evaluate_threat(self, window: List[int], player: int) -> int:
        """Evaluate threat level in a window"""
        score = 0
        empty_count = window.count(0)
        player_count = window.count(player)
        
        if player_count == 3 and empty_count == 1:
            score += 100
        elif player_count == 2 and empty_count == 2:
            score += 30
            
        return score

class MinimaxABAgent(Agent):
    def __init__(self, heuristic_type: HeuristicType = HeuristicType.STANDARD):
        self.heuristic_type = heuristic_type

    def _minimax(self, state: List[List[int]], depth: int, maximizing: bool, 
                 alpha: float, beta: float) -> float:
        valid_moves = [col for col in range(7) if any(state[row][col] == 0 for row in range(6))]

        if not valid_moves:
            raise ValueError("No valid moves left!")
        
        if depth == 0 or not valid_moves or self._is_terminal_node(state):
            return self.evaluate_state(state, 1 if maximizing else 2, self.heuristic_type)

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
        return self._is_winning_state(state, 1) or self._is_winning_state(state, 2)

class DifficultyAgent(MinimaxABAgent):
    def __init__(self, difficulty: str, heuristic_type: HeuristicType = HeuristicType.FIRST_BEST):
        super().__init__(heuristic_type)
        self.difficulty = difficulty.lower()
        self.heuristic_type = heuristic_type

        if self.difficulty == "easy":
            self.max_depth = 3
        elif self.difficulty == "medium":
            self.max_depth = 5
        elif self.difficulty == "hard":
            self.max_depth = 7
        else:
            raise ValueError("Invalid difficulty level. Choose 'easy', 'medium', or 'hard'.")

    def get_chosen_column(self, state: List[List[int]], max_depth: int = None) -> int:
        if max_depth is None:
            max_depth = self.max_depth

        valid_moves = [col for col in range(7) if state[0][col] == 0]
        
        if not valid_moves:
            raise ValueError("No valid moves left!")

        best_score = -math.inf
        best_move = valid_moves[0]
        alpha = -math.inf
        beta = math.inf

        for move in valid_moves:
            new_state = self._make_move(state, move, 1)
            score = self._minimax(new_state, max_depth - 1, False, alpha, beta)
            
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move