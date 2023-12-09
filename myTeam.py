from pacai.agents.capture.capture import CaptureAgent
from pacai.core.directions import Directions
from pacai.core.search.problem import SearchProblem
from pacai.core.distance import manhattan
from pacai.core.search.search import aStarSearch

class OffensiveSearchProblem(SearchProblem):
    def __init__(self, gameState, agent):
        super().__init__()
        self.gameState = gameState
        self.agent = agent

    def startingState(self):
        return self.agent.getCurrentObservation().getAgentPosition(self.agent.index)

    def isGoal(self, state):
        foodList = self.gameState.getBlueFood().asList()
        return len(foodList) == 0

    def successorStates(self, state):
        successors = []
        for action in self.gameState.getLegalActions(self.agent.index):
            successor = self.gameState.generateSuccessor(self.agent.index, action)
            cost = self.gameState.getScore() - successor.getScore()  # You may need to adjust this cost calculation
            successors.append((successor.getAgentPosition(self.agent.index), action, cost))
        return successors

    def actionsCost(self, actions):
        return len(actions)

    def heuristic(self, state, problem=None):
        gameState = self.agent.getCurrentObservation()
        foodList = gameState.getBlueFood().asList()
        capsules = gameState.getBlueCapsules()

        currentPosition = state

        # If there are capsules left, prioritize those.
        if capsules:
            minCapsuleDist = min(self.agent.getMazeDistance(currentPosition, capsule) for capsule in capsules)
            return -gameState.getScore() + minCapsuleDist

        # If no capsules left, go for food.
        if foodList:
            minDistance = min(self.agent.getMazeDistance(currentPosition, food) for food in foodList)
            return -gameState.getScore() + minDistance

        # No food left, just return score.
        return -gameState.getScore()

class DefensiveSearchProblem(SearchProblem):
    def __init__(self, gameState, agent):
        super().__init__()
        self.gameState = gameState
        self.agent = agent

    def startingState(self):
        return self.agent.getCurrentObservation().getAgentPosition(self.agent.index)

    def isGoal(self, state):
        return len(self.gameState.getRedFood().asList()) <= 2

    def successorStates(self, state):
        successors = [(self.gameState.generateSuccessor(self.agent.index, action).getAgentPosition(self.agent.index), action, 1)
                      for action in self.gameState.getLegalActions(self.agent.index)]
        return successors

    def actionsCost(self, actions):
        return len(actions)

    def heuristic(self, state, problem=None):
        gameState = self.agent.getCurrentObservation()
        invaders = [gameState.getAgentState(i) for i in self.agent.getOpponents(gameState) if
                    gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() != None]
        currentPosition = state
        dists = [self.agent.getMazeDistance(currentPosition, a.getPosition()) for a in invaders]

        # If there are invaders, prioritize those.
        if dists:
            return min(dists)

        # If no invaders, stay near own food.
        foodList = gameState.getRedFood().asList()
        if foodList:
            minFoodDist = min(self.agent.getMazeDistance(currentPosition, food) for food in foodList)
            return minFoodDist

        # If no food left, just return 0.
        return 0

class OffensiveAgent(CaptureAgent):
    def chooseAction(self, gameState):
        problem = OffensiveSearchProblem(gameState, self)
        solution = aStarSearch(problem, lambda state, problem=None: problem.heuristic(state))
        

        if solution is None:
            # No solution found, just return some default action.
            return Directions.STOP

        action = solution[0]

        # Avoid enemies who aren't scared.
        for action in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(self.index, action)
            ghosts = [successor.getAgentState(i) for i in self.getOpponents(successor) if not successor.getAgentState(i).isPacman and successor.getAgentState(i).scaredTimer <= 0]
            ghostPositions = [ghost.getPosition() for ghost in ghosts if ghost.getPosition() is not None]
            if ghostPositions and min(self.getMazeDistance(successor.getAgentPosition(self.index), ghost) for ghost in ghostPositions) <= 1:
                continue
            else:
                return action

class DefensiveAgent(CaptureAgent):
    def chooseAction(self, gameState):
        problem = DefensiveSearchProblem(gameState, self)
        solution = aStarSearch(problem, lambda state, problem=None: problem.heuristic(state))


        if solution is None:
            # No solution found, just return some default action.
            return Directions.STOP

        action = solution[0]

        # Stay close to own food.
        myPos = gameState.getAgentPosition(self.index)
        foodList = gameState.getRedFood().asList()
        minFoodDist = min(self.getMazeDistance(myPos, food) for food in foodList)
        if minFoodDist > 2:
            return action

        # Avoid enemies who aren't scared.
        for action in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(self.index, action)
            ghosts = [successor.getAgentState(i) for i in self.getOpponents(successor) if not successor.getAgentState(i).isPacman and successor.getAgentState(i).scaredTimer <= 0]
            ghostPositions = [ghost.getPosition() for ghost in ghosts if ghost.getPosition() is not None]
            if ghostPositions and min(self.getMazeDistance(successor.getAgentPosition(self.index), ghost) for ghost in ghostPositions) <= 1:
                continue
            else:
                return action

def createTeam(firstIndex, secondIndex, isRed,
        first = OffensiveAgent,
        second = DefensiveAgent):
    """
    This function should return a list of two agents that will form the capture team, initialized using firstIndex and secondIndex as their agent indexed. isRed is True if the red team is being created, and will be False if the blue team is being created.
    """

    return [
        first(firstIndex),
        second(secondIndex),
    ]
