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
import sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        """ for reference:        
        if action == 'Stop':
            return 0

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        pacmanPosition = newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        foodList = newFood.asList()

        newGhostStates = successorGameState.getGhostStates()
        # print newGhostStates[0].getPosition()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        sumDistanceFactor = 0
        foodFactor = 1
        ghostIsNear = False
        result = 0
        ghostDistances = []
        for ghost in newGhostStates:
            ghostPosition = ghost.getPosition()
            if ghost.scaredTimer == 0 and abs(pacmanPosition[0] - ghostPosition[0]) <= 3 and abs(pacmanPosition[1] - ghostPosition[1]) <= 3:
                ghostIsNear = True
                distance = util.manhattanDistance(ghostPosition, pacmanPosition)
                ghostDistances.append(distance)

        if ghostIsNear:
            result = min(ghostDistances)
        else:
            if len(foodList) > 0:
                distance, closestFood = min([(manhattanDistance(newPos, food), food) for food in foodList])
                if not distance == 0:
                    result += (1.0/distance)
                else:
                    result += 10

        return result  
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        try:
            fooddist = min([abs(x1-newPos[0]) + abs(y1-newPos[1]) for x1, y1 in newFood.asList()])
        except:
            fooddist = 0
        
        try:
            ghostposition = [x.getPosition() for x in newGhostStates]
            ghostdist = min([abs(x1 - newPos[0]) + abs(y1 - newPos[1]) for x1, y1 in ghostposition])
        except:
            ghostdist = 0
        
        # 要找score最大的: fooddist 越小越好、ghostdist越大越好
        return successorGameState.getScore() - fooddist + ghostdist

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        actions = []

        def max_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)

            v = max([(value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth),
                      action) for action in legal_moves], key=lambda x: x[0])
            actions.append(v[1])
            return v[0]

        def min_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            v = min([value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth)
                     for action in legal_moves])
            return v

        def value(game_state, agent_num=0, depth=1):
            num_agents = game_state.getNumAgents()
            if agent_num >= num_agents:
                if game_state.isLose() or game_state.isWin():
                    return self.evaluationFunction(game_state)
                if self.depth > depth:
                    return value(game_state, 0, depth+1)
                else:
                    return self.evaluationFunction(game_state)

            if agent_num == 0:
                return max_value(game_state, agent_num, depth)
            else:
                return min_value(game_state, agent_num, depth)

        value(gameState)
        return actions[-1]
        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = []

        def max_value(game_state, agent_num, depth, alpha, beta):
            v = (-sys.maxsize, None)
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            for action in legal_moves:
                successor_state = game_state.generateSuccessor(agent_num, action)
                v = max(v, (value(successor_state, agent_num+1, depth, alpha, beta), action), key=lambda x: x[0])
                if v[0] > beta:
                    actions.append(v[1])
                    return v[0]
                alpha = max(alpha, v[0])
            actions.append(v[1])
            return v[0]

        def min_value(game_state, agent_num, depth, alpha, beta):
            v = sys.maxsize
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            for action in legal_moves:
                successor_state = game_state.generateSuccessor(agent_num, action)
                v = min(v, value(successor_state, agent_num+1, depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def value(game_state, agent_num=0, depth=1, alpha=-sys.maxsize, beta=sys.maxsize):
            num_agents = game_state.getNumAgents()
            if agent_num >= num_agents:
                if game_state.isLose() or game_state.isWin():
                    return self.evaluationFunction(game_state)
                if self.depth > depth:
                    return value(game_state, 0, depth+1, alpha, beta)
                else:
                    return self.evaluationFunction(game_state)

            if agent_num == 0:
                return max_value(game_state, agent_num, depth, alpha, beta)
            else:
                return min_value(game_state, agent_num, depth, alpha, beta)

        value(gameState)
        return actions[-1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = []

        def max_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)

            v = max([(value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth),
                      action) for action in legal_moves], key=lambda x: x[0])
            actions.append(v[1])
            return v[0]

        def min_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            # because to choose uniformly at random from their legal moves.            
            v = sum([value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth)
                     for action in legal_moves])/len(legal_moves)
            return v

        def value(game_state, agent_num=0, depth=1):
            num_agents = game_state.getNumAgents()
            if agent_num >= num_agents:
                if game_state.isLose() or game_state.isWin():
                    return self.evaluationFunction(game_state)
                if self.depth > depth:
                    return value(game_state, 0, depth+1)
                else:
                    return self.evaluationFunction(game_state)

            if agent_num == 0:
                return max_value(game_state, agent_num, depth)
            else:
                return min_value(game_state, agent_num, depth)

        value(gameState)
        return actions[-1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

