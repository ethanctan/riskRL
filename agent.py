from game import Game
from itertools import product, permutations
import numpy as np
import random
from consts import TROOP_LIMIT, NUM_PLAYERS
from collections import defaultdict, OrderedDict
from scipy import sparse

class Agent:
    def __init__(self, game_state: dict, agent_id: int): 

        # Initialize game info
        self.agent_id = agent_id
        self.game_counter = 0
        self.game_log = [[game_state]] #list of lists, each list of states corresponds to the state of one game
        self.nodes = game_state["nodes"]
        self.nPlayers = NUM_PLAYERS
        self.edges = game_state["edges"]
        # self.nStates = ((TROOP_LIMIT + 1) * (self.nPlayers + 1)) ** len(self.nodes)
        # self.nActions = 2 * len(game_state["edges"]) + 1 # Each edge (i, j) has 2 actions: i -> j and j -> i. also, the pass action.

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
    
    # Not needed yet
    def get_neighbors(self, node: int):
        neighbors = [node]
        for edge in self.edges:
            if edge[0] == node:
                neighbors.append(edge[1])
            elif edge[1] == node:
                neighbors.append(edge[0])

        return neighbors

    def initialize_R(self):
        print(f"agent {self.agent_id} initializing R")


        TROOP_LOSS_PENALTY_MULTIPLIER = 1
        TERRITORY_LOSS_PENALTY = 10
        TROOP_KILL_REWARD_MULTIPLIER = 1
        TERRITORY_GAIN_REWARD = 10
        ENEMY_KILL_REWARD = 100
        DEATH_PENALTY = 1000

        # the reward matrix assigns rewards to each state-action pair
        # if an action reinforces a node adjacent to an enemy, then the reward is the number of troops added
        # if an action would reduce the number of troops in a territory adjacent to an enemy, then the penalty is the number of troops reduced by
        R = np.zeros((len(self.states), len(self.actions)))

        for state in self.states:
            for node1, node2 in permutations(state, 2):
                (start_node, start_node_owner, start_node_troops) = node1
                (end_node, end_node_owner, end_node_troops) = node2
                action = (start_node, end_node)
                state_index = int(self.states.index(state))
                action_index = int(self.actions.index(action))

                # Check conditions only if start_node_owner is the agent
                if start_node_owner == self.agent_id and end_node_owner != self.agent_id:
                    if start_node_troops >= end_node_troops:
                        # Action would kill an enemy territory if successful
                        R[state_index][action_index] = TERRITORY_GAIN_REWARD
                    else:
                        # Action reduces the number of troops in an enemy territory
                        R[state_index][action_index] = (end_node_troops - start_node_troops) * TROOP_KILL_REWARD_MULTIPLIER
                else: # set reward to 0 for invalid actions
                    R[state_index][action_index] = 0

        return R

        # TODO: Implement rewards that depend on adjacency and how many nodes the enemy owns

    # def initialize_R(self):
    #     print(f"agent {self.agent_id} initializing R")


    #     TROOP_LOSS_PENALTY_MULTIPLIER = 1
    #     TERRITORY_LOSS_PENALTY = 10
    #     TROOP_KILL_REWARD_MULTIPLIER = 1
    #     TERRITORY_GAIN_REWARD = 10
    #     ENEMY_KILL_REWARD = 100
    #     DEATH_PENALTY = 1000

    #     # the reward matrix assigns rewards to each state-action pair
    #     # if an action reinforces a node adjacent to an enemy, then the reward is the number of troops added
    #     # if an action would reduce the number of troops in a territory adjacent to an enemy, then the penalty is the number of troops reduced by
    #     R = defaultdict(dict)

    #     for state in self.states:
    #         for node1, node2 in permutations(state, 2):
    #             (start_node, start_node_owner, start_node_troops) = node1
    #             (end_node, end_node_owner, end_node_troops) = node2
    #             action = (start_node, end_node)

    #             # Check conditions only if start_node_owner is the agent
    #             if start_node_owner == self.agent_id and end_node_owner != self.agent_id:
    #                 if start_node_troops >= end_node_troops:
    #                     # Action would kill an enemy territory if successful
    #                     R[state][action] = TERRITORY_GAIN_REWARD
    #                 else:
    #                     # Action reduces the number of troops in an enemy territory
    #                     R[state][action] = (end_node_troops - start_node_troops) * TROOP_KILL_REWARD_MULTIPLIER
    #             else: # set reward to 0 for invalid actions
    #                 R[state][action] = 0

    #     return R

    #     # TODO: Implement rewards that depend on adjacency and how many nodes the enemy owns

    def initialize_P(self):
        print(f"agent {self.agent_id} initializing P")

        # Flattening the 3D structure to a 2D structure
        rows = len(self.states) * len(self.actions)
        cols = len(self.states)

        # Using a sparse matrix (like lil_matrix for efficient construction)
        P = sparse.lil_matrix((rows, cols))

        # Assuming some logic here to fill in the non-default values.
        # Example:
        # for state_index, action_index, next_state_index, probability in data:
        #     row = state_index * len(self.actions) + action_index
        #     P[row, next_state_index] = probability

        return P

    # def initialize_P(self): #P gives transition probabilities from state to state given action
    #     # each state is a frozenset of tuples where each tuple is (node, owner, troops)
    #     # each action is a tuple (start_node, end_node)
    #     print(f"agent {self.agent_id} initializing P")
    #     # initialize p as np matrix of size (nStates, nActions, nStates) with default probability
    #     P = np.full((len(self.states), len(self.actions), len(self.states)), 1)

    #     return P
    

    def initialize_random_pi(self):
        # each action is a tuple (start_node, end_node)
        # given each state, select a random start node and a random end node
        print(f"agent {self.agent_id} initializing pi")
        pi = np.zeros(len(self.states))
        i = 0
        for state in self.states:
            # pick a random valid action
            state_index = int(self.states.index(state))

            # valid_actions = []
            # nodes_owned_by_agent = []
            # for node in state:
            #     if node[1] == self.agent_id:
            #         nodes_owned_by_agent.append(node[0])

            # for start_node in nodes_owned_by_agent:
            #     for end_node in range(len(self.nodes)):
            #         if start_node != end_node:
            #             valid_actions.append((start_node, end_node))

            # if len(valid_actions) == 0:
            #     pi[state_index] = random.randint(0, len(self.actions) - 1)
            #     continue
                    
            pi[state_index] = np.random.randint(0, len(self.actions) - 1)
            
            if i % 1000 == 0:
                print(f"agent {self.agent_id} initializing pi for state {i}")
            i += 1

        return pi

    # def initialize_random_pi(self):
    #     # each action is a tuple (start_node, end_node)
    #     # given each state, select a random start node and a random end node
    #     print(f"agent {self.agent_id} initializing pi")
    #     pi = {}
    #     for state in self.states:
    #         # pick a random valid action
    #         valid_actions = []
    #         nodes_owned_by_agent = []

    #         for node in state:
    #             if node[1] == self.agent_id:
    #                 nodes_owned_by_agent.append(node[0])

    #         for action in self.actions:
    #             if action[0] in nodes_owned_by_agent:
    #                 valid_actions.append(action)

    #         if len(valid_actions) == 0:
    #             pi[state] = random.choice(self.actions)
    #             continue
                    
    #         pi[state] = random.choice(valid_actions)

    #     return pi
    
    def initialize_states_dict(self):
        # each state will be labelled with a unique frozenset of tuples where each tuple is (node, owner, troops)
        print(f"agent {self.agent_id} initializing states")
        node_ids = range(len(self.nodes))
        owners = range(self.nPlayers + 1)  # Including a 'no owner' state
        troops = range(TROOP_LIMIT + 1)
        print(f"owners: {owners}, troops: {troops}")

        print(f"agent {self.agent_id} generating all possible node states")

        # Generate all possible states for a single node
        single_node_states = list(product(owners, troops))
        print(len(single_node_states))

        print(f"agent {self.agent_id} generating all possible game states")

        # Generate all possible game states
        all_game_states = product(*[single_node_states for _ in node_ids])

        # Create a list with game states as frozensets
        states = []
        i = 0
        for state in all_game_states:
            if i % 100000 == 0:
                print(f"agent {self.agent_id} handling game state {i}")
            i += 1
            game_state = frozenset((node_id,) + state[i] for i, node_id in enumerate(node_ids))
            states.append(game_state)

        print(len(states))

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
    # so we need to turn it into a frozenset in order to be understandable by our agent
    def turn_game_state_into_frozenset(self, game_state: dict):
        # want to make a frozenset of tuples (node_index, owner, troops)
        state_frozenset = []
        for node_index, (node, owner) in enumerate(zip(game_state["nodes"], game_state["owners"])):
            state_frozenset.append((node_index, owner, node))
        return frozenset(state_frozenset)

    def make_move(self):
        current_game_state = self.game_log[self.game_counter][-1]
        current_game_state_frozenset = self.turn_game_state_into_frozenset(current_game_state)

        # get index of action by passing index of state to pi
        current_game_state_index = int(self.states.index(current_game_state_frozenset))
        action = self.actions[int(self.pi[current_game_state_index])]
        self.actions_log[self.game_counter].append(action)
        return action
    
    def approximate_P(self):
        print(f"agent {self.agent_id} approximating P")
        for game_state, action, next_game_state in zip(self.game_log[self.game_counter], self.actions_log[self.game_counter], self.game_log[self.game_counter][1:]):
            game_state_frozenset = self.turn_game_state_into_frozenset(game_state)
            next_game_state_frozenset = self.turn_game_state_into_frozenset(next_game_state)
            game_state_index = int(self.states.index(game_state_frozenset))
            action_index = int(self.actions.index(action))
            next_game_state_index = int(self.states.index(next_game_state_frozenset))
            row = game_state_index * len(self.actions) + action_index
            self.P[row, next_game_state_index] += 1

        # normalize P
        for state_index in range(len(self.states)):
            for action_index in range(len(self.actions)):
                row = state_index * len(self.actions) + action_index
                total = self.P[row, :].sum()
                if total:
                    self.P[row, :] /= total
                print(f"total probability is {total} for state {state_index} and action {action_index}")

        print("P approximated")

    # def approximate_P(self):
    #     print(f"agent {self.agent_id} approximating P")
    #     for game_state, action, next_game_state in zip(self.game_log[self.game_counter], self.actions_log[self.game_counter], self.game_log[self.game_counter][1:]):
    #         game_state_frozenset = self.turn_game_state_into_frozenset(game_state)
    #         next_game_state_frozenset = self.turn_game_state_into_frozenset(next_game_state)
    #         game_state_index = self.states.index(game_state_frozenset)
    #         action_index = self.actions.index(action)
    #         next_game_state_index = self.states.index(next_game_state_frozenset)
    #         self.P[game_state_index][action_index][next_game_state_index] += 1

    #     # normalize P 
    #     for state in self.states:
    #         for action in self.actions:
    #             total = np.sum(self.P[state][action])
    #             if total:
    #                 for next_state in self.states:
    #                     self.P[state][action][next_state] /= total
    #             print(f"total probability is {total} for state {state} and action {action}")

    #     print("P approximated")
            
    def update_pi(self, pi):  
        self.pi = pi

# TODO: Implement foggy and non-foggy

class DynamicProgramming:
    def __init__(self, agent):
        self.agent = agent

    def computeQfromV(self, V):
        print("Computing Q from V")
        # Q's dimensions are (state, action)
        Q = np.zeros((len(self.agent.states), len(self.agent.actions)))
        for state in self.agent.states:
            for a in self.agent.actions:
                state_index = int(self.agent.states.index(state))
                action_index = int(self.agent.actions.index(a))
                expected_value = self.agent.R[state_index, action_index] + np.sum(self.agent.P[state_index, action_index, :] * V)
                Q[state_index][action_index] = expected_value
        return Q

    # def computeQfromV(self, V):
    #     print("Computing Q from V")
    #     # Q's dimensions are (state, action)
    #     Q = defaultdict(lambda: defaultdict(float))
    #     for state in self.agent.states:
    #         for a in self.agent.actions:
    #             expected_value = self.agent.R[state][a]
    #             for next_state, probability in self.agent.P[state][a].items():
    #                 expected_value += probability * V.get(next_state, 0)
    #             Q[state][a] = expected_value
    #     return Q

    def extractMaxPiFromV(self, V):
        print("Extracting max pi from V")
        return np.argmax(self.computeQfromV(V), axis=1)

        # pi = np.zeros(len(self.agent.states))
        # Q = self.computeQfromV(V)
        # for state in Q:
        #     pi[state] = max(Q[state], key=Q[state].get)
        # return pi

    def approxPolicyEvaluation(self, pi, tolerance=0.01):
        print("Approximating policy evaluation")
        epsilon = np.inf
        V = np.zeros(len(self.agent.states))
        i = 0
        while epsilon > tolerance:
            print(f"Approximating policy evaluation iteration {i}")
            nextV = V.copy()
            for s in self.agent.states:
                state_index = int(self.agent.states.index(s))
                Rpis = self.agent.R[state_index, pi[state_index]]
                row = state_index * len(self.agent.actions) + pi[state_index]
                Ppis = self.agent.P[row, :].toarray().flatten()  # Convert sparse row to dense
                nextV[state_index] = Rpis + np.sum(Ppis * V)

            epsilon = np.max(np.abs(nextV - V))
            print(f"epsilon: {epsilon}")
            V = nextV
            i += 1
        return V

    # def approxPolicyEvaluation(self, pi, tolerance=0.01):
    #     print("Approximating policy evaluation")
    #     epsilon = np.inf
    #     V = np.zeros(len(self.agent.states))
    #     i = 0
    #     while epsilon > tolerance:
    #         print(f"Approximating policy evaluation iteration {i}")
    #         nextV = V.copy()
    #         for s in self.agent.states:
    #             state_index = list(self.agent.states.keys()).index(s)
    #             Rpis = self.agent.R[state_index, pi[state_index]]
    #             Ppis = self.agent.P[state_index, pi[state_index], :]  
    #             nextV[state_index] = Rpis + np.sum(Ppis * V)

    #         epsilon = np.max(np.abs(nextV - V))
    #         print(f"epsilon: {epsilon}")
    #         V = nextV
    #         i += 1
    #     return V


    def policyIterationStep(self, pi):
        return self.extractMaxPiFromV(self.approxPolicyEvaluation(pi))

    def policyIteration(self, initial_pi):
        pi = initial_pi.copy()
        iteration_count = 0

        while True:
            next_pi = self.policyIterationStep(pi)
            iteration_count += 1
            print(f"Policy iteration step {iteration_count}")

            if pi == next_pi:
                break
            pi = next_pi

        V = self.approxPolicyEvaluation(pi, 0.1)
        return pi, V, iteration_count