from game import Game

NODELIMIT = 10

class Agent:
    def __init__(self, game: Game, pi: dict):
        self.game = game
        self.nNodes = len(self.game.map.nodes)
        self.nPlayers = len(self.game.players)
        self.nStates = ((NODELIMIT + 1) * (self.nPlayers + 1)) ** self.nNodes
        self.nActions = 2 * len(self.game.map.edges) + 1 # Each edge (i, j) has 2 actions: i -> j and j -> i. also, the pass action.
        self.pi = pi # NOTE: Initialize random policy in the training script
    
    def updateGameState(self, updatedGame: Game):
        self.game = updatedGame

    def approximateP(self):
        pass
        # Naive approach: For each state, look at which states it transitioned to, then update the probabilities

# TODO: Find a way to index the states and actions