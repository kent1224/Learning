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

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    
    
    
    fringe = util.Stack()
    cost = 0
    fringe.push((problem.getStartState(), [], cost))
    visited = []
    
    while not fringe.isEmpty():
        node, action, cost = fringe.pop()
        
        if problem.isGoalState(node):
            return action
        
        if not node in visited:
            for next_node, direction, next_cost in problem.getSuccessors(node):
                fringe.push((next_node, action + [direction], cost + next_cost))
            visited.append(node)
    
    util.raiseNotDefined()
    
    """ 其他參考答案
    fringe = util.Stack()
    fringe.push((problem.getStartState(), [], []))
    
    while not fringe.isEmpty():
        node, action, visited = fringe.pop()
        
        for coord, direction, step in problem.getSuccessors(node):
            if not coord in visited:
                if problem.isGoalState(coord):
                    return action + [direction]
                fringe.push((coord, action + [direction], visited + [node]))
    return []
    """
            

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    cost = 0
    fringe.push((problem.getStartState(), [], cost))
    visited = []
    
    while not fringe.isEmpty():
        node, action, cost = fringe.pop()
        
        if problem.isGoalState(node):
            return action
        
        if not node in visited:
            for next_node, direction, next_cost in problem.getSuccessors(node):
                fringe.push((next_node, action + [direction], cost + next_cost))
            visited.append(node)
    
    util.raiseNotDefined()
        

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    fringe = util.PriorityQueue()
    startstate = problem.getStartState()
    cost = 0
    # ((node, action, cost), priority)
    fringe.push((startstate, [], cost), cost)
    visited = []
    
    while not fringe.isEmpty():
        node, action, cost = fringe.pop()
        
        if problem.isGoalState(node):
            return action
        
        if not node in visited:
            for next_node, direction, next_cost in problem.getSuccessors(node):
                fringe.push((next_node, action + [direction], cost + next_cost), cost + next_cost)
            visited.append(node)
    
    util.raiseNotDefined()

    
    """自己的可行、但觀念不對的答案
    
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), []), 0)
    visited = []
    
    while not fringe.isEmpty():
        node, action = fringe.pop()
        
        for coord, direction, cost in problem.getSuccessors(node):
            if not coord in visited:
                if problem.isGoalState(coord):
                    return action + [direction]
                fringe.push((coord, action + [direction]), problem.getCostOfActions(action + [direction]))
                visited.append(coord)
    return []
    """

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    fringe = util.PriorityQueue()
    startstate = problem.getStartState()
    cost = 0
    # ((node, action, cost), priority)
    fringe.push((startstate, [], cost), cost + heuristic(startstate, problem))
    visited = []
    
    while not fringe.isEmpty():
        node, action, cost = fringe.pop()
        
        if problem.isGoalState(node):
            return action
        
        if not node in visited:
            for next_node, direction, next_cost in problem.getSuccessors(node):
                fringe.push((next_node, action + [direction], cost + next_cost), cost + next_cost + heuristic(next_node, problem))
            visited.append(node)
    
    util.raiseNotDefined()
    
    """自己的錯誤答案    
    
    fringe = util.PriorityQueue()
    start = problem.getStartState()
    fringe.push((start, []), heuristic(start, problem))
    visited = []
    
    while not fringe.isEmpty():
        node, action = fringe.pop()
        
        for coord, direction, cost in problem.getSuccessors(node):
            if not coord in visited:
                new_action = action + [direction]
                if problem.isGoalState(coord):
                    return new_action
                new_cost = problem.getCostOfActions(new_action) + heuristic(coord, problem)
                fringe.push((coord, new_action), new_cost)
    return []
    """


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
