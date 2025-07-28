from abc import ABC, abstractmethod
from collections import deque
import heapq
import numpy as np

class Algorithm(ABC):
    @abstractmethod
    def get_steps(self, initial_state, goal_state):
        pass

    def get_neighbors(self, state):
        state = list(state)
        size = int(len(state) ** 0.5)
        zero_idx = state.index(0)
        zero_row, zero_col = zero_idx // size, zero_idx % size
        
        moves = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = zero_row + dr, zero_col + dc
            if 0 <= new_row < size and 0 <= new_col < size:
                new_idx = new_row * size + new_col
                new_state = state.copy()
                new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
                moves.append((tuple(new_state), new_idx))
        return moves

class BFSAlgorithm(Algorithm):
    def get_steps(self, initial_state, goal_state):
        queue = deque([(initial_state, [])])
        visited = {initial_state}
        
        while queue:
            current_state, path = queue.popleft()
            
            if current_state == goal_state:
                return path
            
            for next_state, action in self.get_neighbors(current_state):
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [action]))
        
        return []

class BestFirstAlgorithm(Algorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic
    
    def get_steps(self, initial_state, goal_state):
        pq = [(self.heuristic(initial_state, goal_state), initial_state, [])]
        visited = {initial_state}
        
        while pq:
            _, current_state, path = heapq.heappop(pq)
            
            if current_state == goal_state:
                return path
            
            neighbors = self.get_neighbors(current_state)
            neighbors.sort(key=lambda x: x[0])
            
            for next_state, action in neighbors:
                if next_state not in visited:
                    visited.add(next_state)
                    h = self.heuristic(next_state, goal_state)
                    heapq.heappush(pq, (h, next_state, path + [action]))
        
        return []

class AStarAlgorithm(Algorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic
    
    def get_steps(self, initial_state, goal_state):
        pq = [(self.heuristic(initial_state, goal_state), 0, initial_state, [])]
        visited = {initial_state: 0}
        
        while pq:
            _, g, current_state, path = heapq.heappop(pq)
            
            if current_state == goal_state:
                return path
            
            if g > visited[current_state]:
                continue
            
            neighbors = self.get_neighbors(current_state)
            neighbors.sort(key=lambda x: x[0])  
            
            for next_state, action in neighbors:
                new_g = g + 1
                if next_state not in visited or new_g < visited[next_state]:
                    visited[next_state] = new_g
                    f = new_g + self.heuristic(next_state, goal_state)
                    heapq.heappush(pq, (f, new_g, next_state, path + [action]))
        
        return []

def hamming_distance(state, goal_state):
    return sum(1 for i, j in zip(state, goal_state) if i != j and i != 0)

def manhattan_distance(state, goal_state):
    size = int(len(state) ** 0.5)
    distance = 0
    for i in range(len(state)):
        if state[i] != 0:
            current_row, current_col = i // size, i % size
            goal_idx = goal_state.index(state[i])
            goal_row, goal_col = goal_idx // size, goal_idx % size
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance
