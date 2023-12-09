"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    no change in discount, and decrease in noise
    """

    answerDiscount = 0.9  # no change the discount factor
    answerNoise = 0.0  # decrease in noise

    return answerDiscount, answerNoise

def question3a():
    """
    """

    answerDiscount = 0.2
    answerNoise = 0.00
    answerLivingReward = -0.8

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    """

    answerDiscount = 0.05
    answerNoise = 0.05
    answerLivingReward = -0.8

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Increase the discount factor to consider distant rewards
    and decrease noise to promote risk-taking.
    """

    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Use high discount to allow the agent to see the high reward in the distance
    and a small negative living reward to prompt quick action.
    """

    answerDiscount = 0.99
    answerNoise = 0.2
    answerLivingReward = 0.01

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    It's impossible to create a policy
    where the agent avoids both exits indefinitely.
    """

    answerDiscount = 0.01
    answerNoise = 0.0
    answerLivingReward = 0.99

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    [Enter a description of what you did here.]
    """

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
