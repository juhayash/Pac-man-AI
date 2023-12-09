"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.priorityQueue import PriorityQueueWithFunction
from pacai.util.priorityQueue import PriorityQueue
from pacai.util.queue import Queue


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    # Initialize the frontier using the initial state of the problem
    frontier = [(problem.startingState(), [])]
    # Initialize the explored set to be empty
    explored = set()

    while frontier:
        # Choose a leaf node and remove it from the frontier
        node, actions = frontier.pop()

        # If the node contains a goal state, return the corresponding solution
        if problem.isGoal(node):
            return actions

        # Add the node to the explored set
        explored.add(node)

        # Expand the chosen node, adding the resulting nodes to the frontier
        # only if not in the frontier or explored set
        for successor, action, _ in problem.successorStates(node):
            new_actions = actions + [action]
            if successor not in explored and (successor, new_actions) not in frontier:
                frontier.append((successor, new_actions))

    # If the frontier is empty, return failure
    return None

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # Create a node with state as the problem's initial state and path cost 0
    node = (problem.startingState(), [])

    # Initialize the frontier as a FIFO queue with node as the only element
    frontier = Queue()
    frontier.push(node)

    # Initialize the explored set as an empty set
    explored = set()

    while not frontier.isEmpty():
        # Choose the shallowest node in the frontier and remove it
        state, actions = frontier.pop()

        # If the node's state is a goal state, return the solution
        if problem.isGoal(state):
            return actions

        # Add the node's state to the explored set if it's not already there
        if state not in explored:
            explored.add(state)

            # Iterate through the successors of the current node
            for successor, action, _ in problem.successorStates(state):
                # Create a child node
                child = (successor, actions + [action])

                # Add the child to the frontier if it's not in explored or frontier
                if child[0] not in explored and not any(
                    item[0] == child[0] for item in frontier.list
                ):
                    frontier.push(child)

    # If the frontier is empty, return failure
    return None

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # Create a node with state as the problem's initial state and path cost 0
    node = (problem.startingState(), [], 0)

    # Initialize the frontier as a priority queue ordered by path cost,
    # with node as the only element
    frontier = PriorityQueue()
    frontier.push(node, 0)

    # Initialize the explored set as an empty set
    explored = set()

    while not frontier.isEmpty():
        # Choose the lowest-cost node in the frontier and remove it
        state, actions, cost = frontier.pop()

        # If the node's state is a goal state, return the solution
        if problem.isGoal(state):
            return actions

        # Add the node's state to the explored set if it's not already there
        if state not in explored:
            explored.add(state)

            # Iterate through the successors of the current node
            for successor, action, step_cost in problem.successorStates(state):
                # Create a child node
                new_cost = cost + step_cost
                child = (successor, actions + [action], new_cost)

                # Add the child to the frontier if it's not in explored or frontier
                in_frontier = any(item[1] == child for item in frontier.heap)

                if not in_frontier and child[0] not in explored:
                    frontier.push(child, new_cost)

                # If the child node is in the frontier with a higher path cost, replace it
                elif in_frontier:
                    for i, item in enumerate(frontier.heap):
                        if item[1] == child and item[0] > new_cost:
                            frontier.heap.remove(item)  # Remove the node with a higher path cost
                            frontier.push(child, new_cost)
                            # Push the updated child node with the lower path cost

    # If the frontier is empty, return failure
    return None

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # Create a node with state as the problem's initial state, path cost 0, and heuristic value
    node = (problem.startingState(), [], 0)

    # Initialize the frontier as a priority queue ordered by the sum of path cost
    # and heuristic value, with node as the only element
    frontier = PriorityQueueWithFunction(lambda x: x[2] + heuristic(x[0], problem))
    frontier.push(node)

    # Initialize the explored set as an empty set
    explored = set()

    while not frontier.isEmpty():
        # Choose the node with the lowest combined cost and heuristic in the frontier and remove it
        state, actions, cost = frontier.pop()

        # If the node's state is a goal state, return the solution
        if problem.isGoal(state):
            return actions

        # Add the node's state to the explored set if it's not already there
        if state not in explored:
            explored.add(state)

            # Iterate through the successors of the current node
            for successor, action, step_cost in problem.successorStates(state):
                # Create a child node
                new_cost = cost + step_cost
                child = (successor, actions + [action], new_cost)

                # Add the child to the frontier if it's not in explored or frontier
                in_frontier = any(item[0] == successor for item in frontier.heap)
                if not in_frontier and successor not in explored:
                    frontier.push(child)

    # If the frontier is empty, return failure
    return None
