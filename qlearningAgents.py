from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util.probability import flipCoin
from pacai.util import reflection
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    one should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # initialize Q-values here.
        self.qValues = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        return self.qValues.get((state, action), 0.0)

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """

        possibleActions = self.getLegalActions(state)

        # If there are no legal actions, return 0.0
        if not possibleActions:
            return 0.0

        return max(self.getQValue(state, action) for action in possibleActions)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """

        possibleActions = self.getLegalActions(state)

        # If there are no legal actions, return None
        if not possibleActions:
            return None

        bestValue = self.getValue(state)
        bestActions = [
            action
            for action in possibleActions
            if self.getQValue(state, action) == bestValue
        ]

        return random.choice(bestActions)

    def getAction(self, state):
        """
        Compute the action to take in the current state.
        With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
        we should take a random action and take the best policy action otherwise.
        """
        # Choose an action by epsilon-greedy algorithm, if there are no legal actions, return None
        possibleActions = self.getLegalActions(state)
        if not possibleActions:
            return None

        # With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`
        # we should take a random action
        if flipCoin(self.getEpsilon()):
            return random.choice(possibleActions)

        # take the best policy action otherwise
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        sample = reward + self.getDiscountRate() * self.getValue(nextState)
        self.qValues[(state, action)] = (
            (1 - self.getAlpha()) * self.getQValue(state, action)
            + self.getAlpha() * sample
        )
class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        self.weights = {}

    def getQValue(self, state, action):
        """
        Should return `Q(state, action) = w * featureVector`,
        where `*` is the dotProduct operator.
        """
        features = self.featExtractor.getFeatures(self, state, action)

        return sum(self.weights.get(feature, 0.0) * value for feature, value in features.items())

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition.
        """

        # Compute the difference between the observed and the expected reward.
        difference = (
            (reward + self.getDiscountRate() * self.getValue(nextState))
            - self.getQValue(state, action)
        )

        # Update the weights.
        features = self.featExtractor.getFeatures(self, state, action)
        for feature, value in features.items():
            self.weights[feature] = (
                self.weights.get(feature, 0.0)
                + self.getAlpha() * difference * value
            )

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        if self.episodesSoFar == self.numTraining:
            print("Weights: ", self.weights)
