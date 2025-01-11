from typing import List, Tuple, Optional
import copy

class GameState:
    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.current_player = 1
        self.last_move = None
        self.moves_history = []

    def make_move(self, column: int) -> bool:
        if column < 0 or column >= self.cols:
            return False
            
        for row in range(self.rows-1, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                self.last_move = (row, column)
                self.moves_history.append(column)
                self.current_player = 3 - self.current_player 
                return True
        return False

    def undo_move(self) -> bool:
        if not self.last_move:
            return False
            
        row, col = self.last_move
        self.board[row][col] = 0
        self.current_player = 3 - self.current_player
        if self.moves_history:
            self.moves_history.pop()
        if len(self.moves_history) > 0:
            # Update last_move to previous move
            last_col = self.moves_history[-1]
            for row in range(self.rows):
                if self.board[row][last_col] != 0:
                    self.last_move = (row, last_col)
                    break
        else:
            self.last_move = None
        return True

    def is_valid_move(self, column: int) -> bool:
        return 0 <= column < self.cols and self.board[0][column] == 0

    def get_valid_moves(self) -> List[int]:
        return [col for col in range(self.cols) if self.is_valid_move(col)]

    def check_winner(self) -> Optional[int]:
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if (self.board[row][col] != 0 and
                    self.board[row][col] == self.board[row][col + 1] == 
                    self.board[row][col + 2] == self.board[row][col + 3]):
                    return self.board[row][col]

        for row in range(self.rows - 3):
            for col in range(self.cols):
                if (self.board[row][col] != 0 and
                    self.board[row][col] == self.board[row + 1][col] == 
                    self.board[row + 2][col] == self.board[row + 3][col]):
                    return self.board[row][col]

        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if (self.board[row][col] != 0 and
                    self.board[row][col] == self.board[row + 1][col + 1] == 
                    self.board[row + 2][col + 2] == self.board[row + 3][col + 3]):
                    return self.board[row][col]

        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if (self.board[row][col] != 0 and
                    self.board[row][col] == self.board[row - 1][col + 1] == 
                    self.board[row - 2][col + 2] == self.board[row - 3][col + 3]):
                    return self.board[row][col]

        return None

    def is_draw(self) -> bool:
        return len(self.get_valid_moves()) == 0 and not self.check_winner()

    def get_state(self) -> List[List[int]]:
        return copy.deepcopy(self.board)

    def load_from_file(self, input_data) -> bool:
        try:
            if isinstance(input_data, str):  # If it's a filename
                with open(input_data, 'r') as f:
                    moves = [int(line.strip()) for line in f if line.strip()]
            elif isinstance(input_data, list):  # If it's a list of moves
                moves = input_data
            else:
                raise ValueError("Input data must be a filename or a list of moves.")

            # Reset the game
            self.__init__()

            # Replay the moves
            for move in moves:
                if not self.make_move(move):
                    return False
            return True
        except (FileNotFoundError, ValueError):
            return False

    def save_to_file(self, filename: str) -> bool:
        try:
            with open(filename, 'w') as f:
                for move in self.moves_history:
                    f.write(f"{move}\n")
            return True
        except IOError:
            return False