from game import Game, Map
from agent import Agent, DynamicProgramming
import random

# We start by trying to train 2 agents on the default map

# Set game parameters.
# numNodes = 9
# edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 5), (1,6), (2, 7), (3, 4), (4, 5), 
#          (5, 6), (6, 7), (7, 4), (4, 8), (5, 8), (6, 8), (7, 8)]
numNodes = 5
edges = [(0, 2), (2, 3), (3, 1), (1, 0), (0, 4), (2, 4), (1, 4)]
# numNodes = 4
# edges = [(0, 1), (1, 3), (3, 2), (2, 0), (0, 3)]
defaultWeight = 1
initialPosition = {
    1: [1],
    2: [2]
}
reinforceAmount = 1
reinforcePlayersOnly = True
killTurn = 100

# Initialize agents
map = Map(numNodes, edges, defaultWeight)
game = Game(map, initialPosition, reinforceAmount, reinforcePlayersOnly)
agent1 = Agent(game.getState(), 1)
agent2 = Agent(game.getState(), 2)

agent1.states = agent1.initialize_states()
agent1.actions = agent1.initialize_actions()
agent1.pi = agent1.initialize_random_pi()
agent1.P = agent1.initialize_P()
agent1.R = agent1.initialize_R()

# agent2.states = agent1.states.copy()
# agent2.actions = agent1.actions.copy()
# agent2.pi = agent1.pi.copy()
# agent2.P = agent1.P.copy()
# agent2.R = agent1.R.copy()

# agent_random = Agent(game.getState(), 2)
# agent_random.states = agent1.states.copy()
# agent_random.actions = agent1.actions.copy()
# agent_random.pi = agent1.pi.copy()
# agent_random.P = agent1.P.copy()
# agent_random.R = agent1.R.copy()

print("All agents initialized. Starting training")

# Training loops
ITERS = 10

#TODO: 10 random + 10 against itself

for game_number in range(ITERS):
    print(f"Game {game_number}")
    # Play game
    for turn_number in range(killTurn):
        print(f"Game {game_number}, turn {turn_number}")

        # Get moves
        move1 = agent1.make_move()
        # move2 = agent2.make_move()

        # make a random move 
        move2_start = random.randint(0, 4)
        move2_end = random.randint(0, 4)

        moves = [(1, move1[0], move1[1]), (2, move2_start, move2_end)]
        # moves = [(1, move1[0], move1[1]), (2, move2[0], move2[1])]
        
        # Update game
        if game.turn(moves):
            break

        # Update agents
        agent1.update_current_game_state(game.getState())
        # agent2.update_current_game_state(game.getState())
        
        print(game.getState())

    # approximate P for each agent
    agent1.approximate_P()
    # agent2.approximate_P()

    # perform policy iteration
    agent1_dp = DynamicProgramming(agent1)
    # agent2_dp = DynamicProgramming(agent2)
    pi1, V1, _ = agent1_dp.policyIteration(agent1.pi)
    # pi2, V2, _ = agent2_dp.policyIteration(agent2.pi)
    agent1.update_pi(pi1)
    # agent2.update_pi(pi2)
    
    # initialize new game and pass to agents
    game = Game(map, initialPosition, reinforceAmount, reinforcePlayersOnly)
    agent1.initialize_new_game(game.getState())
    # agent2.initialize_new_game(game.getState())

# One final game of 20 turns between agent 1 and the player
map = Map(numNodes, edges, defaultWeight)
finalgame = Game(map, initialPosition, reinforceAmount, reinforcePlayersOnly)
for i in range(20):
    print(f"Final game, turn {i}")
    # pass game state to agents
    agent1.update_current_game_state(finalgame.getState())

    # print game state for player
    print(finalgame.getState())

    # Get moves
    move1 = agent1.make_move()
    move_player_start = int(input("Enter your starting node: "))
    move_player_end = int(input("Enter your ending node: "))
    moves = [(1, move1[0], move1[1]), (2, move_player_start, move_player_end)]
    print(f"Agent 1 move: {moves[0][1]} -> {moves[0][2]}")
    print(f"Player move: {moves[1][1]} -> {moves[1][2]}")
    
    # Update game
    if finalgame.turn(moves):
        break

    # Update agents
    agent1.update_current_game_state(finalgame.getState())
    
    print(finalgame.getState())





