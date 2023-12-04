from game import Game
from itertools import product

TROOPLIMIT = 10

class Agent:
    def __init__(self, game: Game, pi: dict):
        self.games = [game]
        self.nodes = self.games[0].map.nodes
        self.nPlayers = len(self.games[0].players)
        self.nStates = ((TROOPLIMIT + 1) * (self.nPlayers + 1)) ** len(self.nodes)
        self.nActions = 2 * len(self.games[0].map.edges) + 1 # Each edge (i, j) has 2 actions: i -> j and j -> i. also, the pass action.
        self.pi = pi # NOTE: Initialize random policy in the training script

        # each state will be labelled with a unique list of tuples where each tuple is (node, owner, troops)
        self.states = {}

    def initialize_states(self):
        node_ids = range(len(self.nodes))
        owners = range(self.nPlayers + 1)  # Including a 'no owner' state
        troops = range(TROOPLIMIT + 1)

        for state_combination in product(product(node_ids, owners, troops), repeat=len(self.nodes)):
            state_label = []
            for node_info in state_combination:
                node_id, owner, troop_count = node_info
                state_label.append((self.nodes[node_id], owner, troop_count))
            self.states[tuple(state_label)] = {}  # Use the state label as key and an empty dict (or any other value) as value

    def approximateP(self):
        for game in self.games:
            log = game.log

        pass
        # Naive approach: For each state, look at which states it transitioned to, then update the probabilities

# TODO: Find a way to index the states and actions