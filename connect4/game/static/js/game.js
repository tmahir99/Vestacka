class Connect4Game {
    constructor() {
        this.board = null;
        this.currentPlayer = 1;
        this.gameType = 'human_vs_human';
        this.difficulty = 'medium';
        this.isGameActive = false;
        this.isThinking = false;
        
        this.initializeElements();
        this.attachEventListeners();
    }
    
    initializeElements() {
        this.boardElement = document.querySelector('.game-board');
        this.gameMessage = document.getElementById('gameMessage');
        this.currentPlayerDisplay = document.getElementById('currentPlayer');
        this.aiThinkingTimeDisplay = document.getElementById('aiThinkingTime');
        this.gameTypeSelect = document.getElementById('gameType');
        this.difficultySelect = document.getElementById('difficulty');
        this.loadGameInput = document.getElementById('loadGame');
        
        // Create the board HTML
        this.boardElement.innerHTML = '';
        for (let row = 0; row < 6; row++) {
            const rowDiv = document.createElement('div');
            rowDiv.className = 'board-row';
            for (let col = 0; col < 7; col++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.col = col;
                rowDiv.appendChild(cell);
            }
            this.boardElement.appendChild(rowDiv);
        }
    }
    
    attachEventListeners() {
        // Board click handler
        this.boardElement.addEventListener('click', (e) => {
            if (!this.isGameActive || this.isThinking) return;
            
            const cell = e.target;
            if (cell.classList.contains('cell')) {
                const column = parseInt(cell.dataset.col);
                this.makeMove(column);
            }
        });
        
        // New game button
        document.getElementById('newGame').addEventListener('click', () => {
            this.gameMessage.textContent = "";
            this.loadGameInput.value = ""
            this.startNewGame();
        });
        
        // Save game button
        document.getElementById('saveGame').addEventListener('click', () => {
            this.saveGame();
        });
        
        // Load game handler
        this.loadGameInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const fileContent = e.target.result;
                    // Split the file content by newlines, filter out empty lines, and convert to integers
                    const dataArray = fileContent
                        .split('\n')
                        .map(item => item.trim())
                        .filter(item => item !== '') // Remove empty strings
                        .map(item => parseInt(item, 10)); // Convert remaining strings to numbers
        
                    console.log(dataArray); // Logs: [0, 0, 1, 3, 2, 2, 3, 2]
                    this.loadGame(dataArray); // Pass the array to your loadGame method
                };
                reader.readAsText(file);
            }
        });        
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Space') {
                this.togglePause();
            } else if (e.key === 'Escape') {
                this.endGame();
            }
        });
    }
    
    async startNewGame() {
        this.gameType = this.gameTypeSelect.value;
        this.difficulty = this.difficultySelect.value;
        
        const response = await fetch('/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                game_type: this.gameType,
                difficulty: this.difficulty
            })
        });
        
        const data = await response.json();
        if (data.board) {
            this.board = data.board;
            this.currentPlayer = data.current_player;
            this.isGameActive = true;
            this.updateDisplay();
            
            if (this.gameType === 'ai_vs_ai') {
                this.startAIGame();
            }
        }
    }
    
    async makeMove(column) {
        if (!this.isGameActive || this.isThinking) return;
        
        const startTime = performance.now();
        this.isThinking = true;
        
        try {
            const response = await fetch('/make_move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ column })
            });
            
            const data = await response.json();
            if (data.error) {
                this.showMessage(data.error);
                return;
            }
            
            this.board = data.board;
            this.currentPlayer = data.current_player;
            
            const endTime = performance.now();
            this.aiThinkingTimeDisplay.textContent = Math.round(endTime - startTime);
            
            this.updateDisplay();
            
            if (data.winner) {
                this.showMessage(`Player ${data.winner} wins!`);
                this.isGameActive = false;
            } else if (data.is_draw) {
                this.showMessage("It's a draw!");
                this.isGameActive = false;
            }
        } finally {
            this.isThinking = false;
        }
    }
    
    async saveGame() {
        if (!this.isGameActive) return;
        
        const response = await fetch('/save_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: 'game_save.txt'
            })
        });
        
        const data = await response.json();
        this.showMessage(data.message || data.error);
    }
    
    async loadGame(content) {
        const response = await fetch('/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                game_type: this.gameType,
                difficulty: this.difficulty,
                load_file: content
            })
        });
        
        const data = await response.json();
        if (data.board) {
            this.board = data.board;
            this.currentPlayer = data.current_player;
            this.isGameActive = true;
            this.updateDisplay();
        }
    }
    
    updateDisplay() {
        const cells = this.boardElement.getElementsByClassName('cell');
        let cellIndex = 0;
        
        for (let row = 0; row < 6; row++) {
            for (let col = 0; col < 7; col++) {
                const cell = cells[cellIndex++];
                cell.classList.remove('player1', 'player2');
                
                if (this.board[row][col] === 1) {
                    cell.classList.add('player1');
                } else if (this.board[row][col] === 2) {
                    cell.classList.add('player2');
                }
            }
        }
        
        this.currentPlayerDisplay.textContent = this.currentPlayer;
    }
    
    showMessage(message) {
        this.gameMessage.textContent = message;
    }
    
    togglePause() {
        if (this.gameType === 'ai_vs_ai') {
            this.isThinking = !this.isThinking;
            if (!this.isThinking) {
                this.startAIGame();
            }
        }
    }
    
    endGame() {
        this.isGameActive = false;
        this.showMessage('Game ended.');
    }
    
    async startAIGame() {
        while (this.isGameActive && !this.isThinking && this.gameType === 'ai_vs_ai') {
            await new Promise(resolve => setTimeout(resolve, 1000));
            await this.makeMove(0); // AI will choose the actual column
        }
    }
}

// Start the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.game = new Connect4Game();
});