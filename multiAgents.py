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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** CS5368 YOUR CODE HERE ***"
        # concentrating on eating. Stay away if a ghost approaches
        newFood = successorGameState.getFood().asList()
        minimumfoodie = float("inf")
        for food in newFood:
            minimumfoodie = min(minimumfoodie, manhattanDistance(newPos, food))

        # Whenever possible, keep your distance from ghosts
        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')
        # reciprocal
        return successorGameState.getScore() + 1.0/minimumfoodie


"""       "Decribe your function:"

    "return successorGameState.getScore()"
"""
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** CS5368 YOUR CODE HERE ***"

         # result format = [score, action]
        result = self.get_value(gameState, 0, 0)

        # Return the result
        return result[1]

    def get_value(self, gameState, index, depth):
        """
        Returns value as pair of [score, action] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """
        # Terminal states:
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(), ""

        # Max-agent: Pacman has index = 0
        if index == 0:
            return self.max_value(gameState, index, depth)

        # Min-agent: Ghost has index > 0
        else:
            return self.min_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = gameState.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # If the successor agent is Pacman, update its depth and index.
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0]

            if current_value > max_value:
                max_value = current_value
                max_action = action

        return max_value, max_action

    def min_value(self, gameState, index, depth):
        """
        Returns the min utility value-action for min-agent
        """
        legalMoves = gameState.getLegalActions(index)
        min_value = float("inf")
        min_action = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # If the successor agent is Pacman, update its depth and index.
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0]

            if current_value < min_value:
                min_value = current_value
                min_action = action

        return min_value, min_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        
        """    
          Returns the minimax action using self.depth and self.evaluationFunction
         "*** YOUR CODE HERE ***" """

        # Format of result = [action, score]
        # Initial state: index = 0, depth = 0, alpha = -infinity, beta = +infinity
        
        return self.maxval(gameState, 0, 0, -float("inf"), float("inf"))[0]

    def alphabeta(self, gameState, agentlocation, depth, alpha, beta):
        """
        Returns the max utility action-score for max-agent with alpha-beta pruning
        """
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        
        
        if agentlocation is 0:
            return self.maxval(gameState, agentlocation, depth, alpha, beta)[1]
        else:
            return self.minval(gameState, agentlocation, depth, alpha, beta)[1]

    def maxval(self, gameState, agentlocation, depth, alpha, beta):
        firstrate = ("max",-float("inf"))
       
       
        for action in gameState.getLegalActions(agentlocation):
            succAction = (action,self.alphabeta(gameState.generateSuccessor(agentlocation,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1, alpha, beta))
            firstrate = max(firstrate,succAction,key=lambda x:x[1])

           
            # Prunning
            if firstrate[1] > beta: return firstrate
            else: alpha = max(alpha,firstrate[1])

        return firstrate

    def minval(self, gameState, agentlocation, depth, alpha, beta):
        firstrate = ("min",float("inf"))
        """
        Returns the min utility action-score for min-agent with alpha-beta pruning
        """

        for action in gameState.getLegalActions(agentlocation):
            succAction = (action,self.alphabeta(gameState.generateSuccessor(agentlocation,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1, alpha, beta))
            firstrate = min(firstrate,succAction,key=lambda x:x[1])

            
            
            # Prunning
            if firstrate[1] < alpha: return firstrate
            else: beta = min(beta, firstrate[1])

        return firstrate
    
     #   "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
    #    util.raiseNotDefined()

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
        "*** CS5368 YOUR CODE HERE ***"
        maxDepth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, "expect", maxDepth, 0)[0]

    def expectimax(self, gameState, action, depth, agentIndex):

        if depth is 0 or gameState.isLose() or gameState.isWin():
            return (action, self.evaluationFunction(gameState))

        # Return the maximum successor value if Pacman (max agent)
        if agentIndex is 0:
            return self.maxvalue(gameState,action,depth,agentIndex)
        # If ghost (EXP agent), return the likelihood.
        else:
            return self.expvalue(gameState,action,depth,agentIndex)

    def maxvalue(self,gameState,action,depth,agentIndex):
        firstrate = ("max", -(float('inf')))
        """
        Returns the max utility value-action for max-agent
        """
        for legalAction in gameState.getLegalActions(agentIndex):
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            succAction = None
        # Update the successor agent's index and depth if it's pacman

            if depth != self.depth * gameState.getNumAgents():
                succAction = action
            else:
                succAction = legalAction
            succValue = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction),
                                        succAction,depth - 1,nextAgent)
            firstrate = max(firstrate,succValue,key = lambda x:x[1])
        return firstrate

    def expvalue(self,gameState,action,depth,agentIndex):
        """
        Returns the max utility value-action for max-agent
        """
        legalActions = gameState.getLegalActions(agentIndex)
        avescore = 0
        propability = 1.0/len(legalActions)
       # Update the successor agent's index and depth if it's pacman   
        for legalAction in legalActions:
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            firstrate = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction),
                                         action, depth - 1, nextAgent)
            avescore += firstrate[1] * propability
        return (action, avescore)

   #     "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
    #    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** CS5368 YOUR CODE HERE ***"

    # Setting up data to be arguments in the evaluation function
    pacman_index = currentGameState.getPacmanPosition()
    ghost_index = currentGameState.getGhostPositions()

    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)
    capsule_count = len(currentGameState.getCapsules())
    near_food = 1

    game_score = currentGameState.getScore()

    # See how far Pacman is from every type of food.
    food_distances = [manhattanDistance(pacman_index, food_position) for food_position in food_list]

    # If there is still food available, set the nearest food value.
    if food_count > 0:
        near_food = min(food_distances)

    # Find how far Pacman is from Ghost (s)
    for ghost_position in ghost_index:
        ghost_distance = manhattanDistance(pacman_index, ghost_position)

         # Prioritize escape if the ghost is too close to Pacman rather than eating the nearby food.
        # simply setting the closest distance to food to a new value.
        if ghost_distance < 2:
            near_food = 99999

    features = [1.0 / near_food, game_score, food_count, capsule_count]

    weights = [10, 200, -100, -10]

    # combination of linear characteristics
    return sum([feature * weight for feature, weight in zip(features, weights)])

# Abbreviation
better = betterEvaluationFunction













