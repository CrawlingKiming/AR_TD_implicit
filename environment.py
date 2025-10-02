import numpy as np
from gymnasium import spaces
import gymnasium as gym 
from numpy import linalg as LA
import random 
from collections import deque
class BoyanChain:
    """
    A variant of the Boyan Chain environment with explicit actions.
    Continous, there is no terminal  (absorbing) state. 
    
    Environment details:
      - States: 0 (terminal) to 12 (starting state) # 13 states 
      - Start state: 12
      - Actions:
          * Action 0: Moves 1 state to the left, reward = 1.
          * Action 2: Moves 2 states to the left, reward = 2.
    """
    def __init__(self, eval_bool=False, policy_action=None):
        self.num_states = 13         # States 0 through 12
        self.start_state = self.num_states - 1  # Starting at state 12
        self.terminal_state = 0      # Terminal state is 0
        self.current_state = self.start_state
        self.is_closed = False 
        self.action_space = spaces.Discrete(2)

        ## Trun on eval status if used for evaluation 
        self.eval = eval_bool 
        if self.eval: 
            assert policy_action is not None
            self.policy_action = policy_action 
            self.stationary_result(self.policy_action)

    def reset(self):
        """
        Resets the environment to the starting state.
        
        Returns:
            int: The starting state.
        """
        self.current_state = self.start_state
        null_reward = None 
        return self.current_state, null_reward

    def reset_initial_state(self):
        self.current_state = self.start_state

    def step(self, action=None):
        """
        Performs one step in the environment based on the chosen action.
        
        Parameters:
            action (int): The action chosen by the agent.
                          Valid actions are 0 (move -2 state, reward = 1)
                          and 1 (move -1 states, reward = 2).
        
        Returns:
            next_state (int): The new state after the action.
            reward (int): The reward for the action (1 for action 0; 2 for action 1).
            done (bool): True if the terminal state (0) is reached.
            info (dict): Additional info (empty in this case).
        """
        # Validate the action.
        if action not in [0, 1] and self.eval:
            """
            For evaluation, 
            should be given with the self.policy_prob 
            """
            s = self.current_state
            if s == self.terminal_state:
                next_state = np.random.choice(self.num_states)
                reward = (self.policy_action[s].copy() + 1) / 2
            elif s == 1:
                next_state = 0
                reward = (self.policy_action[s].copy() + 1) /2
            else:
                action = self.policy_action[s]
                # action 0
                if action == 0:
                    reward = 0.5
                    step_size = 2
                elif action==1:
                    reward = 1
                    step_size = 1
                next_state = s - step_size 
                assert next_state >= 0  
            self.current_state = next_state 
            return reward, next_state
        elif action in [0, 1] :
            if action == 0:
                step_size = 1
                reward = 0.5
            elif action == 1:
                step_size = 2
                reward = 1
        
            if self.current_state == self.terminal_state:
                # Continous Boyd Chain 
                next_state = np.random.choice(self.num_states)
            else:
                step_size = 1 if action == 0 else 2
                next_state = self.current_state - step_size
                if next_state < self.terminal_state:
                    next_state = self.terminal_state
        else: 
            raise ValueError 

        self.current_state = next_state
        # In this design, the environment is continuous, so done is always False.
        done = False
        truncated = None
        what = None
        return next_state, reward, done, truncated, what 

    def close(self):
        """
        Performs cleanup of the environment.
        """
        self.is_closed = True
    
    def stationary_result(self,policy_action):
        """
        Generate transition_prob, if given 
        """
        self.policy_action = policy_action 
        ### get transition probabilit induced by policy_prob 
        N = self.num_states
        P = np.zeros((N, N), dtype=float)
        # fill rows 0 .. N-2
        for i in range(2, N):
            # length: 13, but only 2-12 are used. 
            action = self.policy_action[i] # for amount of p, do step size 2 (action 0)
            if action ==0:
                j1 = i - 2 # two step 
                P[i, j1] = 1.0
            elif action == 1:
                j2 = i - 1 # one step 
                P[i, j2] = 1.0 
            else: 
                raise ValueError 
        P[1, self.terminal_state] = 1.0 
        P[self.terminal_state, :] = 1.0/13

        self.trans_prob = P
        rewards = (self.policy_action.copy() + 1.0)/2
        self.rewards = rewards

    def build_feature_matrix(self, bias, num_states=13, num_features=4):
        """
        Build feature matrix following Boyan(2002)'s setup
        """
        # positions of the basis peaks: 0, 4, 8, 12 for 13 states
        positions = np.linspace(0, num_states - 1, num_features)
        # the width over which each hat drops from 1 to 0
        width = positions[1] - positions[0]
        
        # initialize matrix
        phi = np.zeros((num_states, num_features), dtype=float)
        
        # fill in each entry with the hat basis value
        for i in range(num_states):
            for j, center in enumerate(positions):
                phi[i, j] = max(0.0, 1.0 - abs(i - center) / width)

        phi = phi / np.max(LA.norm(phi, axis=1))
        #print(phi.shape, bias.shape)
        bias = bias[:, None]
        e = np.ones((phi.shape[0], 1))
        phi = np.concatenate([phi, e, bias], axis=1)
        return phi 

class AccessControlEnv(gym.Env):
    """
    The original code is provided by ymzhang, 
    from github.com/ymzhang01/access-control/
    """
    def __init__(self):

        self.n_servers = 10     # No. of servers
        self.priorities = [1/8, 2/8, 4/8, 8/8]  # List of possible priority scores for all customers
        self.n_priorities = len(self.priorities)
        self.free_prob = 0.06   # Probability of server freeing up

        self.observation_space = spaces.Tuple((spaces.Discrete(self.n_servers + 1),
                                               spaces.Discrete(self.n_priorities)))
        self.action_space = spaces.Discrete(2)

        self.busy = None    # Queue containing all busy servers
        self.free = None    # Queue containing all free servers
        self.customer = None    # Priority of current customer
        self.state = None   # State of form [no. free servers, current customer priority]


    def step(self, action):

        if self.state is None:
            raise Exception("Please first initialize state using reset()")

        info = dict()

        # Free up any servers
        if len(self.busy) > 0:
            for _ in range(len(self.busy)):
                if random.random() <= self.free_prob:
                    self.free.append(self.busy.pop())

        if action == 0 or len(self.free) == 0:
            reward = 0
        elif action == 1 and len(self.free) > 0:
            reward = self.customer
            self.busy.append(self.free.pop())

        # Get next customer with equal probabilies 
        self.customer = random.choice(self.priorities)

        # Get next state
        self.state = np.array([len(self.free), self.customer])

        return self.state, reward, False, info


    def reset(self):
        self.customer = random.choice(self.priorities)
        self.busy = deque(maxlen=self.n_servers)
        self.free = deque([1] * self.n_servers, maxlen=self.n_servers)
        self.state = np.array([len(self.free), self.customer])
        return self.state

    def render(self, mode='human'):
        pass
# Example usage:
if __name__ == "__main__":
    env = BoyanChain()
    state = env.reset()
    print("Initial state:", state)
    done = False
    total_reward = 0
