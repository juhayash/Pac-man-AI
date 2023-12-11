"""
This file contains versions of some agents that can be selected to control Pacman.
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

    Methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
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
    A heuristic for the CornersProblem.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

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
    Heuristic for the FoodSearchProblem.

    This heuristic must be consistent to ensure correctness.
    """

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

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    Methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
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
    Implementation of the contest entry.

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
