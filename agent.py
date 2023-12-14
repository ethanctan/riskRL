from game import Game
from itertools import product
import random

TROOPLIMIT = 10

class Agent:
    def __init__(self, game_state: dict): 

        # Initialize game info
        self.game_counter = 0
        self.game_states = [[game_state]] #list of lists, each list of states corresponds to the state of one game
        self.nodes = game_state["nodes"]
        self.nPlayers = len(game_state["owners"])
        self.nStates = ((TROOPLIMIT + 1) * (self.nPlayers + 1)) ** len(self.nodes)
        self.nActions = 2 * len(game_state["edges"]) + 1 # Each edge (i, j) has 2 actions: i -> j and j -> i. also, the pass action.

        # Initialize policy and transition probabilities, and state space
        self.pi = self.initialize_random_pi()
        self.states = self.initialize_states()
        self.P = self.initialize_P()

    def initialize_P(self): #P gives transition probabilities from state to state given action
        # each state is a tuple of tuples where each tuple is (node, owner, troops)
        # each action is a tuple (start_node, end_node)
        P = {}
        for state in self.states:
            P[state] = {}
            for action in range(self.nActions):
                P[state][action] = {}
                for next_state in self.states:
                    P[state][action][next_state] = 0
        return P

    def initialize_random_pi(self):
        # each action is a tuple (start_node, end_node)
        # given each state, select a random start node and a random end node
        pi = {}
        for state in self.states:
            pi[state] = random.randint(0, len(self.nodes) - 1)
        return pi

    def initialize_states(self):
        # each state will be labelled with a unique list of tuples where each tuple is (node, owner, troops)
        node_ids = range(len(self.nodes))
        owners = range(self.nPlayers + 1)  # Including a 'no owner' state
        troops = range(TROOPLIMIT + 1)
        states = {}

        for state_combination in product(product(node_ids, owners, troops), repeat=len(self.nodes)):
            state_label = []
            for node_info in state_combination:
                node_id, owner, troop_count = node_info
                state_label.append((self.nodes[node_id], owner, troop_count))
            states[tuple(state_label)] = {}  # Use the state label as key and an empty dict (or any other value) as value

        return states

    def initialize_new_game(self, game_state: dict):
        self.game_counter += 1
        self.game_states.append([game_state])
        self.nodes = game_state["nodes"]
        self.nPlayers = len(game_state["owners"])
        self.nStates = ((TROOPLIMIT + 1) * (self.nPlayers + 1)) ** len(self.nodes)
        self.nActions = 2 * len(game_state["edges"]) + 1

        self.states = {}
        self.initialize_states()

    def update_current_game_state(self, game_state: dict):
        self.game_states[self.game_counter].append(game_state)

    def make_move(self):
        current_game_state = self.game_states[self.game_counter][-1]
        action = self.pi[current_game_state]
        return action

    def approximate_P(self):
        

        pass
        # Naive approach: For each state, look at which states it transitioned to, then update the probabilities

# TODO: Implement foggy and non-foggy


'''
Our agent needs to play multiple games.
During each game, the agent follows the same policy. 
At the end of each game, the agent updates its estimates of the transition probabilities.
Using this, we perform one step of policy iteration then repeat the process.
'''