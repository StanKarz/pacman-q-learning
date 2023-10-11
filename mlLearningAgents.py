# mlLearningAgents.py
#
# Initially a stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This Q-Learning agent was implemented in Mar. 2023 as part of Coursework 2
# for ML1/PRE module by group "AI", consisting of:
# Stanislav Karzhev (K2257361),
# Joshua O'Hara (K21176641),
# Seamus White (K21220669)
# Innokentii Grigorev (K20073502)
# based on the code by Simon Parsons, Dylan Cope, Lin Li, 2022.


from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        self.pacmanPosition = state.getPacmanPosition()
        self.ghostPositions = state.getGhostPositions()
        self.food = state.getFood()
        self.score = state.getScore()

    def __hash__(self):
        return hash((self.pacmanPosition, tuple(self.ghostPositions), self.food))

    def __eq__(self, other):
        if not isinstance(other, GameStateFeatures):
            return False
        return self.pacmanPosition == other.pacmanPosition and \
               self.ghostPositions == other.ghostPositions and \
               self.food == other.food


class QLearnAgent(Agent):
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.05, gamma: float = 0.8, maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        Initialize Q-learning agent with hyperparameters.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self._old_state = None
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        self.qTable = util.Counter()
        self.countsTable = util.Counter()
        self.wonGames = 0
        self.initialNumPellets = 0
        self.onePelletCollections = 0
        self.old_state = None
        self.old_action = None

    def registerInitialState(self, state: GameState):
        self.initialNumPellets = state.getNumFood()

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self) -> int:
        return self.episodesSoFar

    def getNumTraining(self) -> int:
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        """
        Compute the reward for a given trajectory.

        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        pacman = endState.getPacmanPosition()
        reward = endState.getScore() - startState.getScore()
        ghosts = endState.getGhostPositions()
        pacman = tuple(float(x) for x in pacman)

        for ghost in ghosts:
            if ghost == pacman:
                reward -= 5

        return reward

    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """
        Get the Q-value for a given state and action.

        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.qTable[(state, action)]

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Get the maximum Q-value for a given state.

        Args:
            state: The given state

        Returns:
            The maximum estimated Q-value attainable from the state
        """
        legalActions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        q_values = [self.getQValue(state, action) for action in legalActions]
        return max(q_values)

    def learn(self, state: GameStateFeatures, action: Directions, reward: float, nextState: GameStateFeatures):
        """
        Perform a Q-learning update.

        Args:
            state: The initial state
            action: The action that was took
            nextState: The resulting state
            reward: The reward received on this trajectory
        """
        q_value = self.getQValue(state, action)
        max_q_value_next = self.maxQValue(nextState)
        self.qTable[(state, action)] = q_value + self.alpha * (reward + self.gamma * max_q_value_next - q_value)

    def updateCount(self, state: GameStateFeatures, action: Directions):
        """
        Update the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.countsTable[(state, action)] += 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        """
        Get the number of times an action has been taken in a given state.

        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.countsTable[(state, action)]

    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """

        """Epsilon-greedy exploration strategy, exploration bonus is inversely proportional to the number of times 
         a given action has been taken in the given state """
        return utility + self.epsilon / (1 + counts)

    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to maximize reward while
        balancing gathering data for learning using epsilon-greedy exploration.

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # Get the features of the current state
        stateFeatures = GameStateFeatures(state)

        # Get legal actions and remove the STOP action if it's in the list
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Epsilon-greedy exploration strategy
        if util.flipCoin(self.epsilon):
            # Choose a random action
            return random.choice(legal)
        else:
            # Choose the action with the highest Q-value + exploration bonus
            q_values = [(self.getQValue(stateFeatures, action) +
                         self.explorationFn(self.getQValue(stateFeatures, action),
                                            self.getCount(stateFeatures, action)),
                         action) for action in legal]
            _, action = max(q_values)

        # Generate the next state based on the chosen action
        nextState = state.generatePacmanSuccessor(action)
        nextStateFeatures = GameStateFeatures(nextState)

        # Compute the reward for the transition
        reward = self.computeReward(state, nextState)

        # Update the Q-values and counts
        self.learn(stateFeatures, action, reward, nextStateFeatures)
        self.updateCount(stateFeatures, action)

        self._old_state = state
        self.old_state = stateFeatures
        self.old_action = action
        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        # Print the current game number
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Extract relevant information from the final state
        episode = self.getEpisodesSoFar()
        score = state.getScore()
        pellets_eaten = self.initialNumPellets - state.getNumFood()

        # Update win and one pellet collection counters
        if state.isWin():
            self.wonGames += 1
        if pellets_eaten == 1:
            self.onePelletCollections += 1

        # Print episode summary
        print(f"Episode {episode}: Score = {score}, pellets eaten: {pellets_eaten}")

        # Compute reward based on score difference and update Q-values
        reward = state.getScore() - self._old_state.getScore()
        self.learn(self.old_state, self.old_action, reward, GameStateFeatures(state))

        # Increment the number of episodes completed
        self.incrementEpisodesSoFar()

        # Check if training is complete
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print(f"Number of won games: {self.wonGames}")
            print(f"Number of one pellet collections {self.onePelletCollections}")
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
