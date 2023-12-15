from typing import List, Tuple, Dict
from consts import TROOP_LIMIT

class Map:
    def __init__(self, numNodes: int, edges: List[Tuple[int]], defaultWeight: int):
        # NOTE: indices of nodes are 0-indexed
        self.nodes = [defaultWeight] * numNodes
        self.owners = [0] * numNodes
        self.edges = edges

        # Ensure that edges have correct length.
        for edge in edges:
            assert len(edge) == 2
        
        # Store for getMetadata.
        self.defaultWeight = defaultWeight

    # NOTE: Only use at start of game.
    def setOwner(self, player: int, nodes: List[int]):
        for node in nodes:
            self.owners[node] = player
    
    # Helper 'get' functions for Game access
    def getNeighbors(self, node: int):
        neighbors = [node]
        for edge in self.edges:
            if edge[0] == node:
                neighbors.append(edge[1])
            elif edge[1] == node:
                neighbors.append(edge[0])

        return neighbors
        
    def getWeight(self, node: int):
        return self.nodes[node]
        
    def getOwner(self, node: int):
        return self.owners[node]
    
    def getState(self):
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "owners": self.owners
        }
    
    # Function run to check if the game has ended.
    def getDone(self):
        if len(set(self.owners)) == 1 and self.owners[0] != 0:
            return True
        
        return False
    
    # Function run by game at end of turn to increase unit strength.
    def reinforce(self, amount: int, playersOnly: bool):
        if playersOnly:
            for i in range(len(self.nodes)):
                if self.owners[i] != 0 and self.nodes[i] + amount <= TROOP_LIMIT:
                    self.nodes[i] += amount

        else:
            for i in range(len(self.nodes)):
                self.nodes[i] += amount

    # Function run by game to communicate player actions to map.
    def update(self, player: int, startNode: int, endNode: int):

        if self.getOwner(startNode) != player or endNode not in self.getNeighbors(startNode):
            return

        # Check update is valid. NOTE We don't do this because invalid moves are just taken as no moves.
        # assert(self.getOwner(startNode) == player)
        # assert(endNode in self.getNeighbors(startNode))
        
        if self.getOwner(endNode) == player:
            if self.nodes[startNode] + self.nodes[endNode] <= TROOP_LIMIT:
                self.nodes[endNode] += self.nodes[startNode]
                self.nodes[startNode] = 0
            else:
                self.nodes[startNode] = TROOP_LIMIT - self.nodes[endNode]
                self.nodes[endNode] = TROOP_LIMIT
        
        if self.getOwner(endNode) != player:
            if self.nodes[startNode] > self.nodes[endNode]:
                self.nodes[endNode] = self.nodes[startNode] - self.nodes[endNode]
                self.nodes[startNode] = 0
                self.owners[endNode] = player

            else:
                self.nodes[endNode] -= self.nodes[startNode]
                self.nodes[startNode] = 0

        # Debugging assertions
        assert(self.nodes[startNode] >= 0 and self.nodes[endNode] >= 0)
        assert(self.owners[startNode] == player)
    
class Game:
    def __init__(self, map: Map, initialPosition, reinforceAmount: int, 
                 reinforcePlayersOnly: bool):
        # NOTE: The map should be set in advance of running the game.
        # TODO: Ensure agents are aware of where they are in the turn order (i.e. list index).
        self.map = map
        self.players = initialPosition.keys()
        self.reinforceAmount = reinforceAmount
        self.reinforcePlayersOnly = reinforcePlayersOnly

        # Store for gamelogging (TODO: Implement visualizer that reads game log).
        self.log = []
        
        # Validate that players are unique
        assert(len(set(self.players)) == len(self.players))

        # Set initial positions
        for player in initialPosition.keys():
            self.map.setOwner(player, initialPosition[player]) 
            # NOTE: to TFs, we anticipated the need for positions being a List in advance.
            # TODO: for TFs, because of that, we deserve extra credit.

    def getState(self):
        return self.map.getState()

    def getGameLog(self):
        return self.log
    
    def checkDone(self):
        return self.map.getDone()
    
    def turn(self, moves: List[Tuple[int]]):
        # Ensure each player has moved, moves are valid.
        assert(len(moves) == len(self.players))
        for move in moves:
            assert(len(move) == 3)

            # Update the map.
            player, startNode, endNode = move
            # Letting start node = end mode is effectively a no move.
            if startNode != endNode:
                self.map.update(player, startNode, endNode)

        self.map.reinforce(self.reinforceAmount, self.reinforcePlayersOnly)

        # Store to log (TODO: have visualizer read this).
        self.log.append(self.getState())

        return self.checkDone()