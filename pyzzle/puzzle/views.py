from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from .algorithms import BFSAlgorithm, BestFirstAlgorithm, AStarAlgorithm, hamming_distance, manhattan_distance

@ensure_csrf_cookie
def index(request):
    return render(request, 'puzzle/index.html')

def solve_puzzle(request):
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        initial_state = tuple(data['initial_state'])
        goal_state = tuple(data['goal_state'])  # Now will be [1,2,3,4,5,6,7,8,0]
        algorithm = data['algorithm']
        heuristic = data.get('heuristic', 'manhattan')
        
        heuristic_func = manhattan_distance if heuristic == 'manhattan' else hamming_distance
        
        if algorithm == 'bfs':
            solver = BFSAlgorithm()
        elif algorithm == 'best_first':
            solver = BestFirstAlgorithm(heuristic_func)
        else:  # A*
            solver = AStarAlgorithm(heuristic_func)
        
        steps = solver.get_steps(initial_state, goal_state)
        return JsonResponse({'steps': steps})
    
    return JsonResponse({'error': 'Invalid request method'})