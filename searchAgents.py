"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

# import math
import heapq
import logging


from pacai.core.actions import Actions
from pacai.student.search import aStarSearch
from pacai.core.distance import manhattan
from pacai.core.distance import maze
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core.directions import Directions
# from pacai.util.priorityQueue import PriorityQueue
# from pacai.util.priorityQueue import PriorityQueueWithFunction

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

        # *** Your Code Here ***
        # Initialize the starting state with the starting position and unvisited corners.
        self._startingState = (self.startingPosition, self.corners)

    def startingState(self):
        return self._startingState

    def isGoal(self, state):
        _, unvisited_corners = state
        return len(unvisited_corners) == 0

    def successorStates(self, state):
        currentPosition, unvisited_corners = state
        successors = []

        cardinal_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

        for action in cardinal_directions:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if not hitsWall:
                next_position = (nextx, nexty)
                # Check if the next position is in the unvisited corners and remove it.
                next_unvisited_corners = tuple(
                    corner for corner in unvisited_corners if corner != next_position
                )
                successor = ((next_position, next_unvisited_corners), action, 1)
                successors.append(successor)

        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    # *** Your Code Here ***
    '''
    # he Manhattan distances from the current position to each unvisited corner
    # and returns the sum of these distances.
    # his heuristic does not satisfy the admissibility requirement
    # because it overestimates the cost of reaching the goal.
    currentPosition, unvisited_corners = state

    # If there are no unvisited corners, the heuristic value should be 0.
    if len(unvisited_corners) == 0:
        return 0

    # Calculate the Manhattan distance to each unvisited corner.
    manhattan_distances = []
    for corner in unvisited_corners:
        distance = abs(currentPosition[0] - corner[0])
        + abs(currentPosition[1] - corner[1])
        manhattan_distances.append(distance)

    # Calculate the sum of the minimum distances to each unvisited corner.
    total_distance = sum(manhattan_distances)

    return total_distance
    '''

    '''
    # the minimum distance to the nearest unvisited corner
    # moves to that corner and repeats the process until all corners are visited
    currentPosition, unvisited_corners = state

    if len(unvisited_corners) == 0:
        return 0

    def manhattanDistance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    min_corner_distance = float("inf")
    remaining_corners = list(unvisited_corners)

    while remaining_corners:
        min_distance = min(
            manhattanDistance(currentPosition, corner) for corner in remaining_corners
        )
        nearest_corner = [
            corner
            for corner in remaining_corners
            if manhattanDistance(currentPosition, corner) == min_distance
        ][0]
        min_corner_distance += min_distance
        currentPosition = nearest_corner
        remaining_corners.remove(nearest_corner)

    return min_corner_distance
    '''
    # Useful information.
    corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.
    currentPosition, visitedCorners = state

    # If all corners are visited, return 0
    if visitedCorners == (1, 1, 1, 1):
        return 0

    # Initialize a priority queue for the distances
    distances = []

    # Iterate through each corner
    for i, corner in enumerate(corners):
        # If the corner has not been visited
        if i < len(visitedCorners) and visitedCorners[i] == 0:
            # Calculate the manhattan distance to the unvisited corner
            distance = manhattan(currentPosition, corner)

            # Add the distance and corner index to the priority queue
            heapq.heappush(distances, (distance, i))

    # Initialize total_distance to 0
    total_distance = 0

    # If distances is not empty, start with the nearest unvisited corner
    if distances:
        nearest_distance, nearest_index = heapq.heappop(distances)
        total_distance = nearest_distance

        # Keep track of visited corners as we calculate the heuristic
        new_visitedCorners = list(visitedCorners)
        new_visitedCorners[nearest_index] = 1

        # Continue visiting the remaining unvisited corners
        while distances:
            # Find the next nearest unvisited corner
            next_nearest_distance, next_nearest_index = heapq.heappop(distances)

            # Calculate the distance between the corners
            inter_corner_distance = manhattan(corners[nearest_index], corners[next_nearest_index])

            # Update the total distance and visited corners
            total_distance += inter_corner_distance
            new_visitedCorners[next_nearest_index] = 1
            nearest_index = next_nearest_index

    # Return the total distance as the heuristic
    return total_distance
    # return heuristic.null(state, problem)  # Default to trivial solution

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    # *** Your Code Here **
    '''
    # Manhattan
    # more than 15000

    foodList = foodGrid.asList()

    if not foodList:
        return 0

    # Calculate Manhattan distances to all food positions.
    distances = [abs(position[0] - food[0]) + abs(position[1] - food[1]) for food in foodList]

    # Heuristic: the sum of the shortest and the longest distance to a food.
    return min(distances) + max(distances)
    '''

    '''
    # Minimum spanning tree of the food points
    foodList = foodGrid.asList()

    if not foodList:
        return 0

    # Compute the Manhattan distance between two points.
    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    nodes = [position] + foodList
    unvisited = set(range(1, len(nodes)))
    mst_weight = 0

    # Prim's algorithm.
    def priority_function(item):
        return item[0]

    min_heap = PriorityQueueWithFunction(priority_function)
    min_heap.push((0, 0))  # (distance, node)

    while unvisited and not min_heap.isEmpty():
        dist, current_node = min_heap.pop()

        if current_node in unvisited:
            mst_weight += dist
            unvisited.remove(current_node)

            for neighbor in unvisited:
                min_heap.push((manhattan_distance(nodes[current_node], nodes[neighbor]), neighbor))

    return mst_weight

    # the maximum distance from the current position to any food point
    # 9481 nodes expanded
    foodList = foodGrid.asList()

    if not foodList:
        return 0

    # Compute the Manhattan distance between two points.
    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Find the maximum distance from the current position to any food point.
    max_distance = max(manhattan_distance(position, food) for food in foodList)

    return max_distance

    # combination of the maximum distance to a food point with
    # the toal number of food points
    foodList = foodGrid.asList()

    if not foodList:
        return 0

    # Compute the Manhattan distance between two points.
    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Find the maximum distance from the current position to any food point.
    max_distance = max(manhattan_distance(position, food) for food in foodList)

    # Combine the maximum distance and the total number of food points.
    heuristic_value = max_distance + len(foodList)

    return heuristic_value

    # a heuristic that combines both Manhattan distance and Euclidean distance
    # 9486 nodes expanded
    foodList = foodGrid.asList()

    if not foodList:
        return 0

    # Compute the Manhattan distance between two points.
    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Compute the Euclidean distance between two points.
    def euclidean_distance(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    # Find the maximum average distance from the current position to any food point.
    max_average_distance = max((manhattan_distance(position, food)
    + euclidean_distance(position, food)) / 2 for food in foodList)

    return max_average_distance
    '''
    '''
    # the heuristic to use the minimum of the Manhattan distance
    # and Euclidean distance for each food point.
    # 5524 nodes expanded
    position, foodGrid = state
    foodList = foodGrid.asList()

    if not foodList:
        return 0

    # Compute the Manhattan distance between two points.
    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Compute the Euclidean distance between two points.
    def euclidean_distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # The heuristic will be the sum of the minimum of the Manhattan distance
    # and Euclidean distance for each food point.
    heuristic_value = 0
    for food in foodList:
        heuristic_value += min(
            manhattan_distance(position, food),
            euclidean_distance(position, food)
        )

    return heuristic_value
    '''

    '''
    # 7122 nodes expanded
    position, foodGrid = state
    foodList = foodGrid.asList()

    if not foodList:
        return 0

    def find(parents, x):
        while parents[x] != x:
            x = parents[x]
        return x

    def union(parents, x, y):
        x_root = find(parents, x)
        y_root = find(parents, y)
        parents[x_root] = y_root

    # Compute the minimum spanning tree of the food points using Kruskal's algorithm.
    def mst(foodList):
        edges = PriorityQueue()
        for a in range(len(foodList)):
            for b in range(a + 1, len(foodList)):
                weight = manhattan(foodList[a], foodList[b])
                edges.push((foodList[a], foodList[b]), weight)

        parents = {food: food for food in foodList}
        mst_distance = 0

        while not edges.isEmpty():
            a, b = edges.pop()
            weight = manhattan(a, b)
            if find(parents, a) != find(parents, b):
                mst_distance += weight
                union(parents, a, b)

        return mst_distance

    # Calculate the MST of the food points and
    # add the distance from the position to the closest food.
    min_distance_to_food = min(manhattan(position, food) for food in foodList)
    mst_distance = mst(foodList)

    return min_distance_to_food + mst_distance
    '''

    position, foodGrid = state
    foodList = foodGrid.asList()

    if not foodList:
        return 0

    # Calculate the maze distances between the Pac-Man's position and each food point
    distances = [maze(position, food, problem.startingGameState) for food in foodList]

    # The heuristic will be the maximum of the maze distances
    # between the Pac-Man's position and each food point.
    heuristic_value = max(distances)

    return heuristic_value
    # return heuristic.null(state, problem)  # Default to the null heuristic.

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        # *** Your Code Here ***
        # Define a heuristic function as a lambda function.
        min_food_distance_heuristic = lambda position, problem: min(
            abs(position[0] - food[0]) + abs(position[1] - food[1])
            for food in problem.food.asList()
        )

        # Use A* search with the min_food_distance_heuristic to find the path to the closest dot.
        path = aStarSearch(problem, min_food_distance_heuristic)
        return path

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood()

    def isGoal(self, state):
        x, y = state
        return self.food[x][y]

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        problem = AnyFoodSearchProblem(gameState)
        heuristic = lambda position, problem: min(
            (
                (abs(position[0] - food[0]) + abs(position[1] - food[1]))
                + ((position[0] - food[0])**2 + (position[1] - food[1])**2)**0.5
            )
            / 2
            for food in problem.food.asList()
        )
        path = aStarSearch(problem, heuristic)
        if path:
            return path[0]
        else:
            return Directions.STOP

    def registerInitialState(self, gameState):
        pass
