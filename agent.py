from game import Game
from itertools import product, permutations
import numpy as np
import random
from consts import TROOP_LIMIT, NUM_PLAYERS, DISCOUNT
from collections import defaultdict

class Agent:
    def __init__(self, game_state: dict, agent_id: int): 

        # Initialize game info
        self.agent_id = agent_id
        self.game_counter = 0
        self.game_log = [[game_state]] #list of lists, each list of states corresponds to the state of one game
        self.nodes = game_state["nodes"]
        self.nPlayers = len(game_state["owners"])
        self.edges = game_state["edges"]


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

        TROOP_KILL_REWARD_MULTIPLIER = 1
        TERRITORY_GAIN_REWARD = 10
        TERRITORY_REINFORCE_REWARD_MULTIPLIER = 1
        ENEMY_KILL_REWARD = 100

        # the reward matrix assigns rewards to each state-action pair
        R = defaultdict(dict)

        for state in self.states:
            for node1, node2 in permutations(state, 2):
                (start_node, start_node_owner, start_node_troops) = node1
                (end_node, end_node_owner, end_node_troops) = node2
                action = (start_node, end_node)

                # Check conditions only if start_node_owner is the agent
                if start_node_owner == self.agent_id and end_node_owner != self.agent_id:
                    owners = [node[1] for node in state]
                    if owners.count(end_node_owner) == 1:
                        # Action kills an enemy
                        R[state][action] = ENEMY_KILL_REWARD
                    elif start_node_troops >= end_node_troops:
                        # Action would kill an enemy territory if successful
                        R[state][action] = TERRITORY_GAIN_REWARD
                    else:
                        # Action reduces the number of troops in an enemy territory
                        R[state][action] = (end_node_troops - start_node_troops) * TROOP_KILL_REWARD_MULTIPLIER
                elif start_node_owner == self.agent_id and end_node_owner == self.agent_id:
                    # if an action reinforces a node adjacent to an enemy, then the reward is the number of troops added
                    neighbors = self.get_neighbors(end_node)
                    for neighbor in neighbors:
                        for node in state:
                            if node[0] == neighbor and node[1] != self.agent_id:
                                R[state][action] = start_node_troops * TERRITORY_REINFORCE_REWARD_MULTIPLIER
                                break
                        if R[state].get(action) is not None:
                            break
                        else:
                            R[state][action] = 0
                                
                else: # set reward to 0 for invalid actions
                    R[state][action] = 0

        return R



    def initialize_P(self): #P gives transition probabilities from state to state given action
        # each state is a frozenset of tuples where each tuple is (node, owner, troops)
        # each action is a tuple (start_node, end_node)
        print(f"agent {self.agent_id} initializing P")
        default_prob = 1 / len(self.states)
        P = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: default_prob)))

        return P

    def initialize_random_pi(self):
        # each action is a tuple (start_node, end_node)
        # given each state, select a random start node and a random end node
        print(f"agent {self.agent_id} initializing pi")
        pi = {}
        for state in self.states:
            # pick a random valid action
            valid_actions = []
            nodes_owned_by_agent = []

            for node in state:
                if node[1] == self.agent_id:
                    nodes_owned_by_agent.append(node[0])

            for action in self.actions:
                if action[0] in nodes_owned_by_agent:
                    valid_actions.append(action)

            if len(valid_actions) == 0:
                pi[state] = random.choice(self.actions)
                continue
                    
            pi[state] = random.choice(valid_actions)

        return pi

    def initialize_states(self):
        # each state will be labelled with a unique frozenset of tuples where each tuple is (node, owner, troops)
        print(f"agent {self.agent_id} initializing states")
        node_ids = range(len(self.nodes))
        owners = range(NUM_PLAYERS + 1)  # Including a 'no owner' state
        troops = range(TROOP_LIMIT + 1)

        print(f"agent {self.agent_id} generating all possible node states")

        # Generate all possible states for a single node
        single_node_states = list(product(owners, troops))
        print(len(single_node_states))

        print(f"agent {self.agent_id} generating all possible game states")

        # Generate all possible game states
        all_game_states = product(*[single_node_states for _ in node_ids])

        # Create a dictionary with game states as frozensets
        states = {}
        i = 0
        for state in all_game_states:
            if i % 100000 == 0:
                print(f"agent {self.agent_id} handling game state {i}")
            i += 1
            game_state = frozenset((node_id,) + state[i] for i, node_id in enumerate(node_ids))
            states[game_state] = None  # The value can be anything, e.g., game state score

        return states

    def initialize_new_game(self, game_state: dict):
        self.game_counter += 1
        self.game_log.append([game_state]) #initialize new game state log
        self.actions_log.append([]) #initialize new actions log

    def update_current_game_state(self, game_state: dict):
        self.game_log[self.game_counter].append(game_state)
    
    def turn_game_state_into_frozenset(self, game_state: dict):
        # want to make a frozenset of tuples (node_index, owner, troops)
        state_frozenset = []
        for node_index, (node, owner) in enumerate(zip(game_state["nodes"], game_state["owners"])):
            state_frozenset.append((node_index, owner, node))
        return frozenset(state_frozenset)

    def make_move(self):
        current_game_state = self.game_log[self.game_counter][-1]
        current_game_state_frozenset = self.turn_game_state_into_frozenset(current_game_state)
        action = self.pi[current_game_state_frozenset]
        self.actions_log[self.game_counter].append(action)
        return action
            
    def approximate_P(self):
        print(f"agent {self.agent_id} approximating P")

        for game_state, action, next_game_state in zip(self.game_log[self.game_counter], self.actions_log[self.game_counter], self.game_log[self.game_counter][1:]):
            game_state_frozenset = self.turn_game_state_into_frozenset(game_state)
            next_game_state_frozenset = self.turn_game_state_into_frozenset(next_game_state)
            self.P[game_state_frozenset][action][next_game_state_frozenset] += 1 
            # set all other values in self.P[game_state_frozenset][action] to 0
            for next_state in self.states:
                if next_state != next_game_state_frozenset:
                    self.P[game_state_frozenset][action][next_state] = 0

        # normalize P 
        for state in self.states:
            for action in self.actions:
                total = sum(self.P[state][action].values())
                if total:
                    for next_state in self.states:
                        self.P[state][action][next_state] /= total

        print("P approximated")
        

    def update_pi(self, pi: dict):  
        self.pi = pi


class DynamicProgramming:
    def __init__(self, agent):
        self.agent = agent

    def computeQfromV(self, V):
        print("Computing Q from V")
        Q = defaultdict(lambda: defaultdict(float))
        for state in self.agent.states:
            for a in self.agent.actions:
                expected_value = self.agent.R[state][a]
                for next_state, probability in self.agent.P[state][a].items():
                    expected_value += probability * V.get(next_state, 0)
                Q[state][a] = expected_value
        return Q

    def extractMaxPiFromV(self, V):
        print("Extracting max pi from V")
        pi = defaultdict(lambda: None)
        Q = self.computeQfromV(V)
        for state in Q:
            pi[state] = max(Q[state], key=Q[state].get)
        return pi
    

    def approxPolicyEvaluation(self, pi, tolerance=0.0001):
        print("Approximating policy evaluation")
        epsilon = float('inf')
        V = defaultdict(float)

        i = 0
        while epsilon > tolerance:
            print(f"Approximating policy evaluation iteration {i}")
            nextV = V.copy()
            for s in self.agent.states:
                Rpis = self.agent.R[s][pi[s]]
                Ppis = self.agent.P[s][pi[s]]  # defaultdict
                expected_value = 0
                
                for next_state, probability in Ppis.items():
                    expected_value += (probability * V[next_state])
                
                nextV[s] = Rpis + expected_value * DISCOUNT

            epsilon = max(abs(nextV[s] - V[s]) for s in self.agent.states)
            print(f"epsilon: {epsilon}")
            # print(f"V: {V}")
            V = nextV.copy()
            i += 1
        return V

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