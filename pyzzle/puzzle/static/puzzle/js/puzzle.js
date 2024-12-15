// puzzle/static/puzzle/js/puzzle.js
class PuzzleGame {
    constructor(size = 3) {
        this.size = size;
        this.board = [];
        this.solution = [];
        this.currentStep = 0;
        this.totalSteps = 0;
        this.isPlaying = false;
        this.isSolved = false;
        this.animationFrame = null;
        
        this.setupEventListeners();
    }
    
    initializeBoard() {
        const container = document.getElementById('puzzle-container');
        container.style.gridTemplateColumns = `repeat(${this.size}, 1fr)`;
        
        // Initialize board with correct ordering (1-8, 0)
        this.board = Array.from({length: this.size * this.size - 1}, (_, i) => i + 1);
        this.board.push(0); // Add empty space at the end
        this.shuffleBoard();
        this.renderBoard();
    }
    
    shuffleBoard() {
        // Fisher-Yates shuffle that ensures solvability
        for (let i = this.board.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [this.board[i], this.board[j]] = [this.board[j], this.board[i]];
        }
        
        // Check if puzzle is solvable, if not, swap two non-zero tiles
        if (!this.isSolvable(this.board)) {
            let idx1 = this.board.findIndex(x => x !== 0);
            let idx2 = this.board.findIndex((x, i) => x !== 0 && i > idx1);
            [this.board[idx1], this.board[idx2]] = [this.board[idx2], this.board[idx1]];
        }
    }
    
    isSolvable(board) {
        let inversions = 0;
        for (let i = 0; i < board.length - 1; i++) {
            if (board[i] === 0) continue;
            for (let j = i + 1; j < board.length; j++) {
                if (board[j] === 0) continue;
                if (board[i] > board[j]) inversions++;
            }
        }
        return inversions % 2 === 0;
    }
    
    renderBoard() {
        const container = document.getElementById('puzzle-container');
        container.innerHTML = '';
        
        this.board.forEach((value, index) => {
            const tile = document.createElement('div');
            tile.className = 'puzzle-tile';
            if (value === 0) {
                tile.classList.add('empty');
            } else {
                tile.textContent = value;
                tile.style.backgroundImage = `url('/static/images/image${value}.jpg')`
                tile.style.backgroundSize = 'cover';
            }
            container.appendChild(tile);
        });
        
        // Update step counter
        const stepCounter = document.getElementById('step-counter');
        if (stepCounter) {
            stepCounter.textContent = `Step ${this.currentStep} of ${this.totalSteps}`;
        }
    }
    
    async startGame(algorithm, heuristic) {
        // Hide algorithm selection and show puzzle
        document.getElementById('algorithm-selection').style.display = 'none';
        document.getElementById('game-container').style.display = 'block';
        
        this.initializeBoard();
        await this.solvePuzzle(algorithm, heuristic);
    }
    
    async solvePuzzle(algorithm, heuristic) {
        const csrftoken = this.getCookie('csrftoken');
        
        try {
            const response = await fetch('/solve/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrftoken,
                },
                body: JSON.stringify({
                    initial_state: this.board,
                    goal_state: [...Array(this.size * this.size - 1).keys()].map(x => x + 1).concat([0]),
                    algorithm: algorithm,
                    heuristic: heuristic
                }),
                credentials: 'include'
            });
            
            const data = await response.json();
            this.solution = data.steps;
            this.totalSteps = this.solution.length;
            this.currentStep = 0;
            this.renderBoard();
        } catch (error) {
            console.error('Error solving puzzle:', error);
        }
    }
    
    playSolution() {
        if (!this.isPlaying) return;
        
        if (this.currentStep < this.solution.length) {
            const move = this.solution[this.currentStep];
            this.makeMove(move);
            this.currentStep++;
            this.renderBoard();
            setTimeout(() => this.playSolution(), 500);
        } else {
            this.isPlaying = false;
            this.isSolved = true;
        }
    }
    
    makeMove(tileIndex) {
        const zeroIndex = this.board.indexOf(0);
        [this.board[zeroIndex], this.board[tileIndex]] = [this.board[tileIndex], this.board[zeroIndex]];
    }
    
    setupEventListeners() {
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Enter' && !this.isSolved && this.solution.length > 0) {
                if (!this.isPlaying) {
                    this.isPlaying = true;
                    this.playSolution();
                }
            } else if (e.code === 'Space') {
                e.preventDefault();
                this.isPlaying = !this.isPlaying;
                if (this.isPlaying) {
                    this.playSolution();
                }
            } else if (e.code === 'Escape') {
                window.close();
            }
        });
    }
    
    getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
}