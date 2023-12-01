from game import Map, Game
from typing import List, Tuple, Dict

def playGame(numNodes: int, edges: List[Tuple[int]], defaultWeight: int, initialPosition: Dict[int, List[int]], 
             reinforceAmount: int = 1, reinforcePlayersOnly: bool = True, killTurn: int = 1000):
    # Initialize game
    map = Map(numNodes, edges, defaultWeight)
    game = Game(map, initialPosition, reinforceAmount, reinforcePlayersOnly)
    numPlayers = len(initialPosition)

    for turnIndex in range(killTurn):
        # TODO: Report game state to player
        print(f"Welcome to turn {turnIndex}.")
        print(game.getState())

        moves = []
        for playerIndex in range(numPlayers):
            print(f"Currently talking to player {playerIndex}.")
            startNode = int(input("Enter your movement's starting node: "))
            endNode = int(input("Enter your movement's ending node: "))
            moves.append((playerIndex, startNode, endNode))

        if game.turn(moves):
            break

    print("Here is the final state:")
    print(game.getState())

# NOTE: For user, in initial position, let the player IDs just be 0, 1, 2, ...

# Set game parameters.
numNodes = 5
edges = [(0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (2, 4), (0, 1), (3, 4)]
defaultWeight = 2
initialPosition = {
    0: [0, 1],
    1: [3, 4]
}

playGame(numNodes, edges, defaultWeight, initialPosition)