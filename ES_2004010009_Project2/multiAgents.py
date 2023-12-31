# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        infinity = float('inf')
        ghost_positions = successorGameState.getGhostPositions()

        # run away from ghosts
        for ghost_position in ghost_positions:
            if manhattanDistance(newPos, ghost_position) < 2:
                return -infinity

        # eat food
        num_food = currentGameState.getNumFood()
        new_num_food = successorGameState.getNumFood()
        if new_num_food < num_food:
            return infinity

        min_distance = infinity
        for food in newFood.asList():
            distance = manhattanDistance(newPos, food)
            min_distance = min(min_distance, distance)
        return 1.0 / min_distance


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, game_state, agent=0, depth=0):
        agent %= game_state.getNumAgents()

        if game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        if agent is 0:
            if depth < self.depth:
                return self.maxValue(game_state, agent, depth + 1)
            return self.evaluationFunction(game_state)
        return self.minValue(game_state, agent, depth)

    def maxValue(self, game_state, agent, depth):
        best_value = float("-inf")
        for action in game_state.getLegalActions(agent):
            successor = game_state.generateSuccessor(agent, action)
            v = self.minimax(successor, agent + 1, depth)
            best_value = max(best_value, v)
            if depth == 1 and best_value == v:
                self.action = action
        return best_value

    def minValue(self, game_state, agent, depth):
        best_value = float("inf")
        for action in game_state.getLegalActions(agent):
            successor = game_state.generateSuccessor(agent, action)
            best_value = min(best_value, self.minimax(successor, agent + 1, depth))
        return best_value

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        self.minimax(gameState)
        return self.action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def minimax(self, game_state, agent=0, depth=0, alpha=float("-inf"), beta=float("inf")):
        agent %= game_state.getNumAgents()

        if game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        if agent is 0:
            if depth < self.depth:
                return self.maxValue(game_state, agent, depth + 1, alpha, beta)
            return self.evaluationFunction(game_state)
        return self.minValue(game_state, agent, depth, alpha, beta)

    def maxValue(self, game_state, agent, depth, alpha, beta):
        best_value = float("-inf")
        for action in game_state.getLegalActions(agent):
            successor = game_state.generateSuccessor(agent, action)
            v = self.minimax(successor, agent + 1, depth, alpha, beta)
            best_value = max(best_value, v)
            if depth == 1 and best_value == v:
                self.action = action
            if best_value > beta:
                return best_value
            alpha = max(alpha, best_value)
        return best_value

    def minValue(self, game_state, agent, depth, alpha, beta):
        best_value = float("inf")
        for action in game_state.getLegalActions(agent):
            successor = game_state.generateSuccessor(agent, action)
            best_value = min(best_value, self.minimax(successor, agent + 1, depth, alpha, beta))
            if best_value < alpha:
                return best_value
            beta = min(beta, best_value)
        return best_value

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.minimax(gameState)
        return self.action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, game_state, agent=0, depth=0):
        agent %= game_state.getNumAgents()

        if game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        if agent is 0:
            if depth < self.depth:
                return self.maxValue(game_state, agent, depth + 1)
            return self.evaluationFunction(game_state)
        return self.expValue(game_state, agent, depth)

    def maxValue(self, game_state, agent, depth):
        best_value = float("-inf")
        for action in game_state.getLegalActions(agent):
            successor = game_state.generateSuccessor(agent, action)
            v = self.expectimax(successor, agent + 1, depth)
            best_value = max(best_value, v)
            if depth == 1 and best_value == v:
                self.action = action
        return best_value

    def expValue(self, game_state, agent, depth):
        best_value = 0
        legal_actions = game_state.getLegalActions(agent)
        for action in legal_actions:
            successor = game_state.generateSuccessor(agent, action)
            best_value += self.expectimax(successor, agent + 1, depth)
        return best_value / len(legal_actions)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.expectimax(gameState)
        return self.action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
