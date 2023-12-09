import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        newPos = successorGameState.getPacmanPosition()

        # Calculate the current game score.
        score = successorGameState.getScore()

        # Calculate the distance to the closest food pellet.
        foodList = successorGameState.getFood().asList()
        if len(foodList) > 0:
            minDistance = min([manhattan(newPos, food) for food in foodList])
            score -= minDistance

        # Calculate the distance to the furthest food pellet.
        if len(foodList) > 0:
            maxDistance = max([manhattan(newPos, food) for food in foodList])
            score -= maxDistance

        # Subtract a large number from the score
        # if the distance to the nearest ghost is less than or equal to 2.
        for ghostState in successorGameState.getGhostStates():
            ghostPos = ghostState.getPosition()
            if manhattan(newPos, ghostPos) <= 2:
                score -= 5000

        # Add a score of 1000 if the action results in eating a food pellet.
        if len(foodList) < currentGameState.getFood().count():
            score += 1000

        # Subtract a small number from the score for each remaining food pellet.
        # score -= 10 * len(successorGameState.getFood().asList())

        return score

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Returns the minimax action using self.getTreeDepth() and self.getEvaluationFunction()
        """
        result = self.get_value(state, 0, 0)
        return result[1]

    def get_value(self, state, index, depth):
        """
        Returns value as pair of [score, action] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """
        # Terminal states:
        if len(state.getLegalActions(index)) == 0 or depth == self.getTreeDepth():
            return state.getScore(), ""

        # Max-agent: Pacman has index = 0
        if index == 0:
            return self.max_value(state, index, depth)

        # Min-agent: Ghost has index > 0
        else:
            return self.min_value(state, index, depth)

    def max_value(self, gameState, index, depth):
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = gameState.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""

        for action in legalMoves:
            if action == Directions.STOP:
                continue
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0]

            if current_value > max_value:
                max_value = current_value
                max_action = action

        return max_value, max_action

    def min_value(self, state, index, depth):
        """
        Returns the min utility value-action for min-agent
        """
        legalMoves = state.getLegalActions(index)
        min_value = float("inf")
        min_action = ""

        for action in legalMoves:
            if action == Directions.STOP:
                continue
            successor = state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0]

            if current_value < min_value:
                min_value = current_value
                min_action = action

        return min_value, min_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def max_value(self, state, depth, alpha, beta):
        """
        Compute the max value of the minimax tree.
        """
        if state.isWin() or state.isLose() or depth == self.getTreeDepth():
            return self.getEvaluationFunction()(state)
        value = float('-inf')
        for action in state.getLegalActions(self.index):
            successor_state = state.generateSuccessor(self.index, action)
            value = max(value, self.min_value(successor_state, depth, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value

    def min_value(self, state, depth, alpha, beta):
        """
        Compute the min value of the minimax tree.
        """
        if state.isWin() or state.isLose() or depth == self.getTreeDepth():
            return self.getEvaluationFunction()(state)
        value = float('inf')
        for i in range(state.getNumAgents()):
            if i == self.index:
                continue
            for action in state.getLegalActions(i):
                successor_state = state.generateSuccessor(i, action)
                value = min(value, self.max_value(successor_state, depth + 1, alpha, beta))
                if value <= alpha:
                    return value
                beta = min(beta, value)
                if beta <= alpha:
                    break
        return value

    def getAction(self, state):
        """
        Compute the minimax action using alpha-beta pruning.
        """
        alpha = float('-inf')
        beta = float('inf')
        value = float('-inf')
        best_action = None
        for action in state.getLegalActions(self.index):
            successor_state = state.generateSuccessor(self.index, action)
            min_val = self.min_value(successor_state, 0, alpha, beta)
            if min_val > value:
                value = min_val
                best_action = action
            if value >= beta:
                return best_action
            alpha = max(alpha, value)
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the expectimax action from the current gameState.
        """
        # Initialize variables
        agentIndex = self.index
        depth = 0
        bestAction = None
        v = float("-inf")

        # Evaluate each legal action
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            newValue = self.expectimax(
                successor, (agentIndex + 1) % gameState.getNumAgents(),
                depth
            )
            if newValue > v:
                v = newValue
                bestAction = action

        return bestAction

    def expectimax(self, gameState, agentIndex, depth):
        """
        Returns the expectimax value of the current game state for a particular agent.
        """
        if gameState.isWin() or gameState.isLose() or depth == self.getTreeDepth():
            return self.getEvaluationFunction()(gameState)

        if agentIndex == self.index:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.expValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        """
        Returns the max value of the current game state for a particular agent.
        """
        v = float("-inf")
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = max(
                v, self.expectimax(
                    successor,
                    (agentIndex + 1) % gameState.getNumAgents(),
                    depth
                )
            )
        return v

    def expValue(self, gameState, agentIndex, depth):
        """
        Returns the expected value of the current game state for a particular agent.
        """
        v = 0
        actions = gameState.getLegalActions(agentIndex)
        probability = 1.0 / len(actions)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v += probability * self.expectimax(
                successor,
                (agentIndex + 1) % gameState.getNumAgents(),
                depth + 1
            )
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()

    foodList = currentGameState.getFood().asList()
    foodCount = len(foodList)
    capsuleCount = len(currentGameState.getCapsules())
    closestFoodDistance = 1

    score = currentGameState.getScore()
    foodDistance = [manhattan(pacmanPosition, foodPos) for foodPos in foodList]

    if foodCount > 0:
        closestFoodDistance = min(foodDistance)

    for ghostPosition in ghostPositions:
        ghostDistance = manhattan(pacmanPosition, ghostPosition)
        if ghostDistance < 2:
            closestFoodDistance = 1500

    features = [1.0 / closestFoodDistance, score, foodCount, capsuleCount]
    weights = [10, 200, -100, -10]

    return sum([feature * weight for feature, weight in zip(features, weights)])

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
