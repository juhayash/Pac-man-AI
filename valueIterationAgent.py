from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # Compute the values here.
        # raise NotImplementedError()
        # Initialize the value of all states to zero.
        for state in self.mdp.getStates():
            self.values[state] = 0.0

        # Run the value iteration process.
        for _ in range(self.iters):
            newValues = self.values.copy()

            for state in self.mdp.getStates():
                qValues = []

                for action in self.mdp.getPossibleActions(state):
                    qValues.append(self.getQValue(state, action))

                if len(qValues) > 0:
                    newValues[state] = max(qValues)

            self.values = newValues

    def getQValue(self, state, action):
        """
        Compute the Q-value of action in state from the value function stored in self.values.
        """

        qValue = 0.0

        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discountRate * self.values[nextState])

        return qValue

    def getPolicy(self, state):
        """
        Compute the best action to take in a state.
        """

        possibleActions = self.mdp.getPossibleActions(state)

        # If there are no legal actions, return None.
        if len(possibleActions) == 0:
            return None

        bestAction = possibleActions[0]
        maxQValue = self.getQValue(state, bestAction)

        for action in possibleActions:
            qValue = self.getQValue(state, action)
            if qValue > maxQValue:
                bestAction = action
                maxQValue = qValue

        return bestAction

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
