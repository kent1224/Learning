# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:13:50 2017

@author: 14224
"""

import util

def depthFirstSearch(problem): 
# def breadthFirstSearch(problem):
# def uniformCostSearch(problem):
# def aStarSearch(problem, heuristic=nullHeuristic):

    
    fringe = util.Stack() # DFS: LIFO
    #fringe = util.Queue()  # BFS: FIFO
    #fringe = util.PriorityQueue() # UCS, A*S: pop depends on cost
    
    cost = 0
    startstate = problem.getStartState()
    fringe.push((startstate, [], cost)) # DFS, BFS: (state, action, cost)
    # fringe.push((startstate, [], cost), cost) # UCS: ((state, action, cost), priority)
    # fringe.push((startstate, [], cost), cost + heuristic(startstate, problem)) # A*S: ((state, action, cost), priority)
    visited = []
    
    while not fringe.isEmpty():
        node, action, cost = fringe.pop()
        
        if problem.isGoalState(node):
            return action
        
        if not node in visited:
            for next_node, direction, next_cost in problem.getSuccessors(node):
                fringe.push((next_node, action + [direction], cost + next_cost)) #DFS, BFS
                # fringe.push((next_node, action + [direction], cost + next_cost), cost + next_cost) # UCS
                # fringe.push((next_node, action + [direction], cost + next_cost), cost + next_cost + heuristic(next_node, problem)) # A*S
            visited.append(node)