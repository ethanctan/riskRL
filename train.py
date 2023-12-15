from game import Game, Map
from agent import Agent, DynamicProgramming

# We start by trying to train 2 agents on the default map

# Set game parameters.
# numNodes = 9
# edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 5), (1,6), (2, 7), (3, 4), (4, 5), 
#          (5, 6), (6, 7), (7, 4), (4, 8), (5, 8), (6, 8), (7, 8)]
# numNodes = 5
# edges = [(1, 3), (3, 4), (4, 2), (2, 1), (1, 5), (3, 5), (2, 5)]
numNodes = 4
edges = [(1, 2), (2, 4), (4, 3), (3, 1), (1, 4)]
defaultWeight = 2
initialPosition = {
    1: [2],
    2: [3]
}
reinforceAmount = 1
reinforcePlayersOnly = True
killTurn = 1000

# Initialize agents
map = Map(numNodes, edges, defaultWeight)
game = Game(map, initialPosition, reinforceAmount, reinforcePlayersOnly)
agent1 = Agent(game.getState(), 1)
agent2 = Agent(game.getState(), 2)

print("All agents initialized. Starting training")

# Training loops
ITERS = 10
for game_number in range(ITERS):
    # Play game
    for turn_number in range(killTurn):
        print(f"Game {game_number}, turn {turn_number}")
        # pass game state to agents
        agent1.update_current_game_state(game.getState())
        agent2.update_current_game_state(game.getState())

        # Get moves
        move1 = agent1.make_move()
        move2 = agent2.make_move()
        moves = [(1, move1[0], move1[1]), (2, move2[0], move2[1])]
        
        # Update game
        if game.turn(moves):
            break

        # Update agents
        agent1.update_current_game_state(game.getState())
        agent2.update_current_game_state(game.getState())
        
        print(game.getState())

    # approximate P for each agent
    agent1.approximate_P()
    agent2.approximate_P()

    # perform policy iteration
    agent1_dp = DynamicProgramming(agent1)
    agent2_dp = DynamicProgramming(agent2)
    pi1, V1, _ = agent1_dp.policyIteration(agent1.pi)
    pi2, V2, _ = agent2_dp.policyIteration(agent2.pi)
    agent1.update_pi(pi1)
    agent2.update_pi(pi2)
    
    # initialize new game and pass to agents
    game = Game(map, initialPosition, reinforceAmount, reinforcePlayersOnly)
    agent1.initialize_new_game(game.getState())
    agent2.initialize_new_game(game.getState())



