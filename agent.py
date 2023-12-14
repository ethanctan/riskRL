from game import Game
from itertools import product, combinations
import random

TROOP_LIMIT = 10

class Agent:
    def __init__(self, game_state: dict, agent_id: int): 

        # Initialize game info
        self.agent_id = agent_id
        self.game_counter = 0
        self.game_log = [[game_state]] #list of lists, each list of states corresponds to the state of one game
        self.nodes = game_state["nodes"]
        self.nPlayers = len(game_state["owners"])
        self.edges = game_state["edges"]
        # self.nStates = ((TROOP_LIMIT + 1) * (self.nPlayers + 1)) ** len(self.nodes)
        # self.nActions = 2 * len(game_state["edges"]) + 1 # Each edge (i, j) has 2 actions: i -> j and j -> i. also, the pass action.
        self.states = self.initialize_states()
        self.actions = self.initialize_actions()

        # Initialize policy and transition probabilities, and state space
        self.pi = self.initialize_random_pi()
        self.P = self.initialize_P()
        self.R = self.initialize_R()

        # Initialize log for self actions
        self.actions_log = [[]]

    def initialize_actions(self):
        # each action is a tuple (start_node, end_node)
        actions = []
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i != j:
                    actions.append((i, j))
        return actions
    
    def getNeighbors(self, node: int):
        neighbors = [node]
        for edge in self.edges:
            if edge[0] == node:
                neighbors.append(edge[1])
            elif edge[1] == node:
                neighbors.append(edge[0])

        return neighbors

    def initialize_R(self):


        TROOP_LOSS_PENALTY_MULTIPLIER = 1
        TERRITORY_LOSS_PENALTY = 10
        TROOP_KILL_REWARD_MULTIPLIER = 1
        TERRITORY_GAIN_REWARD = 10
        ENEMY_KILL_REWARD = 100
        DEATH_PENALTY = 1000

        # the reward matrix assigns rewards to each state-action pair
        # if an action reinforces a node adjacent to an enemy, then the reward is the number of troops added
        # if an action would reduce the number of troops in a territory adjacent to an enemy, then the penalty is the number of troops reduced by
        R = {}
        for state in self.states:
            node_pairs = list(combinations(state, 2)) # all possible actions
            for node_pair in node_pairs:
                start_node = action[0][0]
                start_node_owner = action[0][1]
                start_node_troops = action[0][2]
                end_node = action[1][0]
                end_node_owner = action[1][1]
                end_node_troops = action[1][2]
                action = (start_node, end_node)

                # if an action would kill an enemy territory if successful, then the reward is 10
                if start_node_owner == self.agent_id and end_node_owner != self.agent_id and start_node_troops >= end_node_troops:
                    R[state][action] = TERRITORY_GAIN_REWARD

                # if an action reduces the number of troops in an enemy territory, the reward is the number of troops reduced by
                elif start_node_owner == self.agent_id and end_node_owner != self.agent_id and start_node_troops < end_node_troops:
                    R[state][action] = (end_node_troops - start_node_troops) * TROOP_KILL_REWARD_MULTIPLIER

                # TODO: Implement rewards that depend on adjacency and how many nodes the enemy owns


    def initialize_P(self): #P gives transition probabilities from state to state given action
        # each state is a set of tuples where each tuple is (node, owner, troops)
        # each action is a tuple (start_node, end_node)
        P = {}
        for state in self.states:
            P[state] = {}
            for action in self.actions:
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
        # each state will be labelled with a unique set of tuples where each tuple is (node, owner, troops)
        node_ids = range(len(self.nodes))
        owners = range(self.nPlayers + 1)  # Including a 'no owner' state
        troops = range(TROOP_LIMIT + 1)
        states = {}

        for state_combination in product(product(node_ids, owners, troops), repeat=len(self.nodes)):
            state_label = []
            for node_info in state_combination:
                node_id, owner, troop_count = node_info
                state_label.append((self.nodes[node_id], owner, troop_count))
            # create set of tuples from the list of tuples we have
            state_label = set(state_label)
            states[state_label] = 0
        return states

    def initialize_new_game(self, game_state: dict):
        self.game_counter += 1
        self.game_log.append([game_state]) #initialize new game state log
        self.actions_log.append([]) #initialize new actions log

    def update_current_game_state(self, game_state: dict):
        self.game_log[self.game_counter].append(game_state)
    
    # each game state is the following format: 
    '''
    {
            "nodes": self.nodes,
            "edges": self.edges,
            "owners": self.owners
        }
    '''
    # so we need to turn it into a set in order to be understandable by our agent
    def turn_game_state_into_set(self, game_state: dict):
        # want to make a set of tuples (node_index, owner, troops)
        state_set = []
        for node_index, node, owner in enumerate(zip(game_state["nodes"], game_state["owners"])):
            set.append((node_index, owner, node))
        return set(state_set)

    def make_move(self):
        current_game_state = self.game_log[self.game_counter][-1]
        current_game_state_set = self.turn_game_state_into_set(current_game_state)
        action = self.pi[current_game_state_set]
        self.actions_log[self.game_counter].append(action)
        return action
            
    def approximate_P(self, P: list):
        for game_state, action, next_game_state in zip(self.game_log[self.game_counter], self.actions_log[self.game_counter], self.game_log[self.game_counter][1:]):
            game_state_set = self.turn_game_state_into_set(game_state)
            next_game_state_set = self.turn_game_state_into_set(next_game_state)
            self.P[game_state_set][action][next_game_state_set] += 1      

        # normalize P 
        for state in self.states:
            for action in self.actions:
                total = sum(self.P[state][action].values())
                if total != 0:
                    for next_state in self.states:
                        self.P[state][action][next_state] /= total
        
        return P
        # Naive approach: For each state and action, look at which states it transitioned to, then update the probabilities


# TODO: Implement foggy and non-foggy


'''
Our agent needs to play multiple games.
During each game, the agent follows the same policy. 
At the end of each game, the agent updates its estimates of the transition probabilities.
Using this, we perform one step of policy iteration then repeat the process.
'''