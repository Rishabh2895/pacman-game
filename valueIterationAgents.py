# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        states = mdp.getStates()

        # Write value iteration code here
        "*** CS5368 YOUR CODE HERE ***"

        # Iterating over Numbers
        # appling the value iteration formula using probab and state values and then saving the maximum state value

        # Value iteration loop set up for self.iterations iterations
        for i in range(iterations):
          #intializing a temporary counter to store an iteration's value for each state.
          iterationValues = util.Counter()
          #looking at each state
          for s in self.mdp.getStates():
            # if the state is terminal, the reward is the exit reward and no discounted rewards as it is the absorbing state
            if self.mdp.isTerminal(s):
              self.values[s] = self.mdp.getReward(s, 'exit', '')
            # if the state is non-terminal, then finding the best value as the maximum of expected sum of rewards of different actions.
            else:
              actions = self.mdp.getPossibleActions(s)
              iterationValues[s] = max([self.computeQValueFromValues(s, a) for a in actions])
          self.values = iterationValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** CS5368 YOUR CODE HERE ***"

        #Computing Qvalue by using the value saved in above fuction in self.values
        #Initialing QValue as Zero

        qvalue = 0

        for nextState, probab in self.mdp.getTransitionStatesAndProbs(state, action):
            qvalue += probab * (self.mdp.getReward(state, action, nextState) + (self.discount * self.values[nextState]))
        return qvalue

        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** CS5368 YOUR CODE HERE ***"

        #Iterating over each and every action and computing the value
        #maximum value will be our best action or policy

        if not self.mdp.getPossibleActions(state):
            return None
        policy = util.Counter()

        for action in self.mdp.getPossibleActions(state):
            for nextState, probab in self.mdp.getTransitionStatesAndProbs(state, action):
                policy[action] += probab * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])

        return policy.argMax()

        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
