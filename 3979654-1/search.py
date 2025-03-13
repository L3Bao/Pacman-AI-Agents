# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    stack = util.Stack()    # initializing a new stack
    startState = problem.getStartState()    # get the start state
    stack.push((startState, []))    # put the startState on the top of the stack
    visited = set()     # a set of all the visited states

    while not stack.isEmpty():
        currentState, actions = stack.pop()     # Get the first element of the stack
        if problem.isGoalState(currentState):   # If the current state is the goal state, print the actions made to get there
            # for i in actions:
            return actions
        
        if currentState not in visited:         # If the current state has not been visited, add it to the visited set
            visited.add(currentState)

            for successor, action, stepCost in problem.getSuccessors(currentState):
                # print(action, stepCost, end=" ")
                if successor not in visited:
                    newPath = actions + [action]
                    stack.push((successor, newPath))

    util.raiseNotDefined()
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    startState = problem.getStartState()
    queue.push((startState, []))
    visited = set()

    while not queue.isEmpty():
        currentState, actions = queue.pop()

        if problem.isGoalState(currentState):
            for i in actions:
                print(i, end=" ")
            return actions

        if currentState not in visited:
            visited.add(currentState)

            for successor, action, stepCost in problem.getSuccessors(currentState):
                if successor not in visited:
                    newPath = actions + [action]
                    queue.push((successor, newPath))

    util.raiseNotDefined()
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue

    # If the start state is the goal, return an empty path
    if problem.isGoalState(problem.getStartState()):
        return []

    # Initialize the priority queue
    priorityQueue = PriorityQueue()
    startState = problem.getStartState()
    priorityQueue.push((startState, [], 0), 0)  # (state, path, cost)

    # Dictionary to track visited states and their least cost
    visited = {}

    while not priorityQueue.isEmpty():
        # Pop the node with the smallest cumulative cost
        currentState, actions, currentCost = priorityQueue.pop()

        # Check if we've reached the goal state
        if problem.isGoalState(currentState):
            return actions

        # If the state has not been visited or a cheaper path is found
        if currentState not in visited or currentCost < visited[currentState]:
            visited[currentState] = currentCost  # Mark state as visited with its cost

            # Explore all successors
            for successor, action, stepCost in problem.getSuccessors(currentState):
                newCost = currentCost + stepCost
                newActions = actions + [action]
                print(action, stepCost, end=" ")
                # Push the successor to the priority queue
                priorityQueue.push((successor, newActions, newCost), newCost)

    util.raiseNotDefined()
    # Return an empty path if no solution is found
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []

    priorityQueue = util.PriorityQueue()
    startState = problem.getStartState()
    priorityQueue.push((startState, [], 0), 0)

    visited = {}

    while not priorityQueue.isEmpty():

        currentState, actions, currentCost = priorityQueue.pop()

        if problem.isGoalState(currentState):
            return actions

        if currentState not in visited or currentCost < visited[currentState]:
            visited[currentState] = currentCost

            for successor, action, stepCost in problem.getSuccessors(currentState):
                newCost = currentCost + stepCost
                newActions = actions + [action]
                # print(action, stepCost, end=" ")
                heuristicCost = heuristic(successor, problem)
                totalCost = newCost + heuristicCost

                priorityQueue.push((successor, newActions, newCost), totalCost)
    util.raiseNotDefined()

    return []


#####################################################
# EXTENSIONS TO BASE PROJECT
#####################################################

# Extension Q1e
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


#####################################################
# Abbreviations
#####################################################
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
