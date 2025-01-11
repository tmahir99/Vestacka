from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .game_logic import GameState
from .agents import DifficultyAgent

game_states = {}

def index(request):
    return render(request, 'game/index.html')

@csrf_exempt
def save_game(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        filename = data.get('filename', 'game_save.txt')
        session_key = request.session.session_key
        
        if not session_key or session_key not in game_states:
            return JsonResponse({'error': 'No active game'}, status=400)
            
        game = game_states[session_key]['game']
        if game.save_to_file(filename):
            return JsonResponse({'message': 'Game saved successfully'})
        else:
            return JsonResponse({'error': 'Failed to save game'}, status=400)

@csrf_exempt
def new_game(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        game_type = data.get('game_type', 'human_vs_human')
        difficulty = data.get('difficulty', 'medium')
        load_file = data.get('load_file', None)
        
        game = GameState()
        
        if load_file:
            game.load_from_file(load_file)
        
        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key
        
        agent = None
        if game_type != 'human_vs_human':
            agent = DifficultyAgent(difficulty)
        
        game_states[session_key] = {
            'game': game,
            'game_type': game_type,
            'difficulty': difficulty,
            'agent': agent
        }
        
        return JsonResponse({
            'board': game.get_state(),
            'current_player': game.current_player,
            'game_type': game_type
        })

@csrf_exempt
def make_move(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        column = data.get('column')
        session_key = request.session.session_key
        
        if not session_key or session_key not in game_states:
            return JsonResponse({'error': 'No active game'}, status=400)
            
        game_data = game_states[session_key]
        game = game_data['game']
        
        if not game.make_move(column):
            return JsonResponse({'error': 'Invalid move'}, status=400)
            
        winner = game.check_winner()
        is_draw = game.is_draw()
        
        ai_move = None
        if (not winner and not is_draw and 
            game_data['game_type'] in ['human_vs_ai', 'ai_vs_ai']):
            
            agent = game_data['agent']
            valid_moves = game.get_valid_moves()
            print(f"valid AI moves: {valid_moves}")
            if agent.get_chosen_column(game.get_state()) in valid_moves:
                ai_move = agent.get_chosen_column(game.get_state())
            else:
                ai_move = 12
            game.make_move(ai_move)
            winner = game.check_winner()
            is_draw = game.is_draw()
        
        return JsonResponse({
            'board': game.get_state(),
            'current_player': game.current_player,
            'winner': winner,
            'is_draw': is_draw,
            'ai_move': ai_move
        })
