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
    node = (problem.getStartState(), []) # My representaion of a node
    visited = set()
    fringe = util.Stack()
    fringe.push(node)

    while fringe:
        node = fringe.pop()
        if problem.isGoalState(node[0]): return node[1]
        if node[0] in visited:
            continue
        visited.add(node[0])
        for successor in problem.getSuccessors(node[0]):
            ls = [] + node[1]
            ls.append(successor[1])
            childNode = (successor[0], ls)
            if childNode[0] not in visited:
                fringe.push(childNode)
    return None
                    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    node = (problem.getStartState(), []) # My representaion of a node
    visited = set()
    fringe = util.Queue()
    fringe.push(node)
    
    while fringe:
        node = fringe.pop()
        if problem.isGoalState(node[0]): return node[1]
        if node[0] in visited:
            continue
        visited.add(node[0])
        for successor in problem.getSuccessors(node[0]):
            ls = [] + node[1]
            ls.append(successor[1])
            childNode = (successor[0], ls)
            if childNode[0] not in visited:
                fringe.push(childNode)
    return None


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    node = (problem.getStartState(), []) # My representaion of a node
    visited = set()
    fringe = util.PriorityQueueWithFunction(lambda plan: problem.getCostOfActions(plan[1]))
    fringe.push(node)
    while fringe:
        node = fringe.pop()
        if problem.isGoalState(node[0]): return node[1]
        if node[0] in visited:
            continue

        visited.add(node[0])
        for successor in problem.getSuccessors(node[0]):
            ls = [] + node[1]
            ls.append(successor[1])
            childNode = (successor[0], ls)
    
            if childNode[0] not in visited:
                fringe.push(childNode)

    return None


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    node = (problem.getStartState(), []) # My representaion of a node
    visited = set()
    fringe = util.PriorityQueueWithFunction(lambda plan: problem.getCostOfActions(plan[1]) 
        + heuristic(plan[0], problem))
    fringe.push(node)
    while fringe:
        node = fringe.pop()
        if problem.isGoalState(node[0]): return node[1]
        if node[0] in visited:
            continue
            
        visited.add(node[0])
        for successor in problem.getSuccessors(node[0]):
            ls = [] + node[1]
            ls.append(successor[1])
            childNode = (successor[0], ls)
    
            if childNode[0] not in visited:
                fringe.push(childNode)

    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

def __generalUniformSearch__(problem, strat="bfs", heuristic=nullHeuristic):
   """
   finds the path according to the problem using a uniformed search algorithm
   :param problem:
   :param strat: strategy used (the algorithm)
       options:    dfs, bfs, ucs
   :return:
   """

   """
   UniformSearch(G,v)   ( v is the vertex where the search starts )
       fringe S := {};   ( start with an empty stack )
       for each vertex u, set visited[u] := false;
       push S, v;
       while (S is not empty) do
           u := pop S;
           if isGoalState(u):
               return path to u
           if (not visited[u]) then
              visited[u] := true;
              for each unvisited neighbour w of u
                 push S, w;
           end if
       end while
   """
   start_time = time.time()

   structures = {
       "dfs": util.Stack(),
       "bfs": util.Queue(),
       "ucs": util.PriorityQueueWithFunction(lambda plan: problem.getCostOfActions(plan[1])),
       "astar": util.PriorityQueueWithFunction(
           lambda plan: problem.getCostOfActions(plan[1]) + heuristic(plan[0], problem)
       )
   }

   fringe = structures[strat]

   # format of a plan: (state: tuple, actions: list, cost)
   current = (problem.getStartState(), tuple(), 0)
   fringe.push(current)

   visited = set()

   def wasVisited(plan):
       return visited.__contains__(plan[0])

   while not fringe.isEmpty():
       # expanding
       current = fringe.pop()

       if problem.isGoalState(current[0]):
           print("[%s]:    completed in %f sec, plan: %s" %
                 (strat, (time.time() - start_time), current))
           # print("Visited nodes: ", visited)
           return __listify__(current[1])

       # here is where the tie-breaking should happen, sort the successors in a certain way,
       # this is what affects which nodes go first

       if not wasVisited(current):  # push nonvisited nodes and mark this node as visited
           successors = problem.getSuccessors(current[0])
           # print("Current plan: {}\nSuccessors:   {}\n".format(current, successors))

           # pushing to the fringe
           visited.add(current[0])
           unvisited = [unv for unv in successors if not wasVisited(unv)]
           for s in unvisited:  # the unvisited successors
               fringe.push(__extendPlan__(current, s))

   print("WARNING: No path found")
   return ["Stop"]

def __listify__(x):
   """wraps with a list, if not already a list"""
   return [] if x is None else \
           x if type(x) is list else list(x)

def __tuplify__(x):
   """wraps with a list, if not already a list"""
   return tuple([]) if x is None else \
           x if type(x) is tuple else tuple([x])

def __extendPlan__(currentPlan, succPlan):
   """
   given a current plan state (which will have all steps to reach this state)
   and a successorPlan (which will contain the next steps),
   returns the joint plan
   format of a plan: (state, actions, cost)
       state: tuple of length 2
       actions:  list containing the needed moves to reach the state
       cost:   integer representing the cost

   Example output:
       ((5, 5), ('Stop', 'West', 'East'), 2)

   :param currentPlan:
   :param succPlan: (state, actions, cost)
   :return: returns a tuple in the form (state, actions, cost)
   """

   actions = tuple(list(__tuplify__(currentPlan[1])) + list(__tuplify__(succPlan[1])))
   extendedPlan = succPlan[0], actions, (currentPlan[2] + succPlan[2])

   # print("Extended {} + {} = {}".format(currentPlan, succPlan, extendedPlan))
   return extendedPlan

def __getFringeString__(fringe):
   if fringe is None:
       raise Exception("Fringe is None")
   return "[\n\t{}\n]".format("\n\t".join([str(x) for x in fringe.list]))
