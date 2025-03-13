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

# import logging

# Configure logging
# logging.basicConfig(
#     filename='output_log.txt',    # Log file name
#     level=logging.DEBUG,          # Set the logging level (DEBUG captures all logs)
#     format='%(asctime)s - %(levelname)s - %(message)s',)

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

        
    # Base score
        score = successorGameState.getScore()

    # 1. Ghost avoidance and scared ghost chasing
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        for i, distance in enumerate(ghostDistances):
            if distance <= 1 and newScaredTimes[i] == 0:  # Active ghost nearby
                return score - 1000  # Heavy penalty for danger
            if newScaredTimes[i] > 0 and distance <= 3:  # Scared ghost within range
                score += 200 / (distance + 1)  # Reward chasing scared ghosts

    # 2. Food proximity
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            closestDistance = min(foodDistances)
            score += 10 / (closestDistance + 1)  # Reward for closer food

    # 3. Stop action penalty
        if action == Directions.STOP:
            score -= 100
            
    # 4. Discourage oscilating between 2 position
        if newPos == currentGameState.getPacmanPosition():
            score -= 70  # Penalize returning to the same position

    # Run with frameTime 1 to see better
        # print("Score: ", score)
        # print("Action: ", action)
        return score

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
        def minimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            if agentIndex == 0:     # agent is pacman
                return maxVal(agentIndex, depth, state)
            else:
                return minVal(agentIndex, depth, state)
            
        def maxVal(agentIndex, depth, state):
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            
            return max(
                minimax(1, depth, state.generateSuccessor(agentIndex, action))
                for action in actions
            )
            
        def minVal(agentIndex, depth, state):
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
                
            nextAgent = agentIndex + 1
            if nextAgent >= state.getNumAgents():
                nextAgent = 0
                depth += 1
                
            return min(
                minimax(nextAgent, depth, state.generateSuccessor(agentIndex, action))
                for action in actions
            )
        
        legalMoves = gameState.getLegalActions(0)  # Pacman's legal actions
        bestAction = max(
            legalMoves,
            key=lambda action: minimax(1, 0, gameState.generateSuccessor(0, action)),
        )
        return bestAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """



    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(agentIndex, depth, state, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                print(f"Evaluating node at depth {depth}: {state}")
                return self.evaluationFunction(state)
            
            if agentIndex == 0:
                return maxVal(agentIndex, depth, state, alpha, beta)
            else:
                return minVal(agentIndex, depth, state, alpha, beta)
            
        def maxVal(agentIndex, depth, state, alpha, beta):
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            
            v = float('-inf')
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                logging.debug(f"Max: Exploring action {action}, depth {depth}")
                v = max(v, alphaBeta(1, depth, successor, alpha, beta))
                if v >= beta:
                    logging.debug(f"Max: Pruning at action {action}, v={v}, beta={beta}")
                    return v
                alpha = max(alpha, v)
            return v
        
        def minVal(agentIndex, depth, state, alpha, beta):
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            
            v = float('inf')
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                depth += 1
                
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                logging.debug(f"Min: Exploring action {action}, depth {depth}")
                v = min(v, alphaBeta(nextAgent, depth, successor, alpha, beta))
                if v <= alpha:
                    logging.debug(f"Min: Pruning at action {action}, v={v}, alpha={alpha}")
                    return v
                beta = min(beta, v)
            return v
    
    
        alpha = float('-inf')
        beta = float('inf')
        legalMoves = gameState.getLegalActions(0)

        bestAction = None
        bestScore = float('-inf')

        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = alphaBeta(1, 0, successor, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction    
            
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0: 
                return maxValue(agentIndex, depth, state)
            else: 
                return expValue(agentIndex, depth, state)

        def maxValue(agentIndex, depth, state):
            actions = state.getLegalActions(agentIndex)
            if not actions: 
                return self.evaluationFunction(state)

            v = float('-inf')
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, expectimax(1, depth, successor))
            return v

        def expValue(agentIndex, depth, state):
            actions = state.getLegalActions(agentIndex)
            if not actions: 
                return self.evaluationFunction(state)

            total = 0
            probability = 1 / len(actions)  # Uniform distribution for ghost actions
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():  # Last ghost
                nextAgent = 0
                depth += 1  # Increase depth after all agents' turns

            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                total += probability * expectimax(nextAgent, depth, successor)
            return total

        # Pacman's turn (agentIndex = 0)
        legalMoves = gameState.getLegalActions(0)
        bestAction = None
        bestScore = float('-inf')

        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(1, 0, successor)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    # 1. Food-related score
    foodDistances = [manhattanDistance(pacmanPos, f) for f in food]
    if foodDistances:  # Avoid division by zero
        closestFood = min(foodDistances)
        score += 10 / (closestFood + 1)  # Encourage getting closer to food

    # 2. Ghost-related score
    for ghost in ghosts:
        ghostPos = ghost.getPosition()
        ghostDistance = manhattanDistance(pacmanPos, ghostPos)

        if ghost.scaredTimer > 0:  # Scared ghost
            score += 200 / (ghostDistance + 1)  # Encourage chasing scared ghosts
        elif ghostDistance <= 1:  # Active ghost nearby
            score -= 1000  # Heavy penalty for being near an active ghost

    # 3. Capsule-related score
    capsuleDistances = [manhattanDistance(pacmanPos, c) for c in capsules]
    if capsuleDistances:
        closestCapsule = min(capsuleDistances)
        score += 20 / (closestCapsule + 1)  # Encourage moving toward capsules

    # 4. Remaining food penalty
    score -= len(food) * 5  # Penalize for more remaining food

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
