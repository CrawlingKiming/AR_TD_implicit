import gymnasium as gym
import itertools
import matplotlib
import numpy as np

import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from sklearn.kernel_approximation import RBFSampler

import argparse
import os 
import pickle
from environment import AccessControlEnv

# Wrapper used for Pendulum
class CustomInfinitePendulumWrapper(gym.Wrapper):
    """
    A wrapper for the pendulum
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        # Take a step in the base environment.
        observation, reward, terminated, truncated, info = self.env.step(action)

        new_reward = reward
        new_reward = new_reward / 16.27 # scales the reward value 
        # Force the episode not to end by setting terminated and truncated to False.
        return observation, new_reward, False, info
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs
# ---------------------------
# Hand-Coded Estimator Class
# ---------------------------
class Estimator():
    """
    A linear function approximator for Q(s,a) using RBF features.
    Maintains one weight vector per action.
    """
    def __init__(self, action_set, num_actions=None):
        self.action_set = action_set 
        if num_actions is None:
            self.nA = env.action_space.n
        else: 
            self.nA = num_actions 
        example_state = env.reset()
        # feature, \phi 
        self.num_features = len(self.featurize_state(example_state))
        # Initialize one weight vector per action.
        # corresponds to \theta_0
        self.weights = [np.random.uniform(-0.5, 0.5, size=self.num_features)for _ in range(self.nA)] #np.zeros(self.num_features) 
        # corresonds to \omega_0
        self.omega = 0.0 
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        featurized = featurizer.transform([state])[0]
        return featurized
    
    def predict(self, state, a_idx=None):
        """
        a_idx represents the index
        If action a is provided, returns Q(s,a); else returns Q(s,a) for all actions.
        """
        phi = self.featurize_state(state)
        if a_idx is not None:
            return np.dot(self.weights[a_idx], phi)
        else:
            return np.array([np.dot(self.weights[i], phi) for i in range(self.nA)])

# ---------------------------
# Epsilon-Greedy Policy
# ---------------------------
def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Returns a function that takes an observation and returns a probability distribution
    over actions using an epsilon-greedy policy.
    """
    def policy_fn(observation):
        # A : array of probabilities 
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action_idx = np.argmax(q_values)
        A[best_action_idx] += (1.0 - epsilon) # if epsilon is 0.0, becomes greedy policy
        return A
    return policy_fn

# ---------------------------
# SARSA with Standard Average-Reward TD (lambda) Update
# ---------------------------
def sarsa_vanilla(env_dict, estimator, num_episodes, c_alpha=1.0,
                  epsilon=0.1, epsilon_decay=1.0, alpha0=0.1, sanity_every=1, R = 1000, max_step=100):
    """
    Implements SARSA using the standard TD update under AR
    """
    
    nA = estimator.nA#env.action_space.n
    stats = EpisodeStats(episode_omega=np.zeros(num_episodes),
                         episode_rewards=np.zeros(num_episodes))
    val1_list = []
    val2_list = []
    state = env.reset() # initialize 
    z = 0.0

    for i_episode in range(num_episodes): # number of iterations 

        alpha_eff=step_sizes[i_episode] # current step size 
        if i_episode < thr:
            current_epsilon = 0.25
        elif i_episode < thr*2:
            current_epsilon = 0.125
        else:
            current_epsilon = 0.00

        policy = make_epsilon_greedy_policy(estimator, current_epsilon, nA) # returns array of action probabilites (for epsilon-greedy)
        action_idx = np.random.choice(nA, p=policy(state))
        action = estimator.action_set[action_idx]

        new_state, reward, terminated, _ = env.step(action)
        new_action_idx = np.random.choice(nA, p=policy(new_state)) # next action 
        new_action = estimator.action_set[new_action_idx]

        stats.episode_rewards[i_episode] += reward
        stats.episode_omega[i_episode] += estimator.omega 
        phi_state = estimator.featurize_state(state)
        q_sa = np.dot(estimator.weights[action_idx], phi_state)

        if terminated:
            # if terminated, we reset the environment
            new_state = env.reset()
            phi_next = estimator.featurize_state(new_state)
            q_next = np.dot(estimator.weights[new_action_idx], phi_next)
        else:
            phi_next = estimator.featurize_state(new_state)
            q_next = np.dot(estimator.weights[new_action_idx], phi_next)
        # comupte \delta_t = R_t^{\mu} - \omega_t + \phi(s_{t+1}^{\mu}, a_{t+1}^{mu})\theta_t - \phi(s_{t}^{\mu}, a_{t}^{\mu})\theta_t
        td_target = reward - estimator.omega + q_next
        td_error = td_target - q_sa
        # update z 
        z = Lambda * z + phi_state
        # update parameters
        estimator.weights[action_idx] += (alpha_eff * td_error * z) 
        estimator.omega = estimator.omega + c_alpha * alpha_eff* (reward - estimator.omega) 
        state = new_state

        if (i_episode) % sanity_every == 0:
            # for code sanity checks, not used, all values incorporated in stats object
            val1, val2 = 0,0
            val1_list.append(val1)
            val2_list.append(val2)
        print("\rStandard TD SARSA Episode {}/{} - α_eff: {:.4f} - AR_{:.4g}".format(
            i_episode+1, num_episodes, alpha_eff, estimator.omega), end="")
        sys.stdout.flush()
    print("\nStandard TD SARSA training finished.")
    return stats, np.array(val1_list), np.array(val2_list)

# ---------------------------s
# SARSA with Average-Reward Implicit TD(lambda) Update
# ---------------------------
def sarsa_implicit(env_dict, estimator, num_episodes, c_alpha=1.0,
                   epsilon=0.1, epsilon_decay=1.0, alpha0=0.5, sanity_every=1, R = 1000, max_step=100):
    """
    Implements SARSA using the implicit TD update under AR
    """
    nA = estimator.nA#env.action_space.n
    stats = EpisodeStats(episode_omega=np.zeros(num_episodes),
                         episode_rewards=np.zeros(num_episodes))
    val1_list = []
    val2_list = []
    z = 0.0 
    state = env.reset()
    thr = 5000  # epsilon value decreasing for every thr iteration 
    current_epsilon =0.25
    policy = make_epsilon_greedy_policy(estimator, current_epsilon, nA)
    action_idx = np.random.choice(nA, p=policy(state))
    
    for i_episode in range(num_episodes): 

        alpha_eff=step_sizes[i_episode] # current step size 
        if i_episode < thr:
            current_epsilon = 0.25
        elif i_episode < thr*2:
            current_epsilon = 0.125
        else:
            current_epsilon = 0.00 # reaches zero eventually 
        policy = make_epsilon_greedy_policy(estimator, current_epsilon, nA)
        action = estimator.action_set[action_idx]
        new_state, reward, terminated, _ = env.step(action)
        new_action_idx = np.random.choice(nA, p=policy(new_state))
        new_action = estimator.action_set[new_action_idx]
        stats.episode_rewards[i_episode] += reward
        stats.episode_omega[i_episode] += estimator.omega
        phi_state = estimator.featurize_state(state)
        q_sa = np.dot(estimator.weights[action_idx], phi_state)

        if terminated:
            new_state = env.reset()
            phi_next = estimator.featurize_state(new_state)
            q_next = np.dot(estimator.weights[new_action_idx], phi_next)
        else:
            phi_next = estimator.featurize_state(new_state)
            q_next = np.dot(estimator.weights[new_action_idx], phi_next)
        # comupte \delta_t = R_t^{\mu} - \omega_t + \phi(s_{t+1}^{\mu}, a_{t+1}^{mu})\theta_t - \phi(s_{t}^{\mu}, a_{t}^{\mu})\theta_t
        td_target = reward - estimator.omega + q_next
        td_error = td_target - q_sa
        # update z 
        z = Lambda * z + phi_state
        # update parameters 
        estimator.weights[action_idx] += (alpha_eff * td_error * z) / (1 + alpha_eff * np.linalg.norm(z)**2)
        estimator.omega = estimator.omega + c_alpha * alpha_eff* (reward - estimator.omega) / (1+ c_alpha * alpha_eff)

        # performs projection if R (projection radius) is given
        if R:
            if np.linalg.norm(estimator.weights[action_idx]) > R - 1:
                estimator.weights[action_idx] = (
                    estimator.weights[action_idx] 
                    / np.linalg.norm(estimator.weights[action_idx]) 
                    * (R - 1)
                )
            if np.linalg.norm(estimator.omega) > 1:
                estimator.omega = (
                    estimator.omega 
                    / np.linalg.norm(estimator.omega) 
                    * 1.0
                )

        state = new_state
        action_idx = new_action_idx

        if (i_episode) % sanity_every == 0:
            # for code sanity checks, not used, all values incorporated in stats. object
            val1, val2 = 0,0
            val1_list.append(val1)
            val2_list.append(val2)
        print("\rImplicit TD SARSA Episode {}/{} - α_eff: {:.5f} - AR_{:.4f}".format(
            i_episode+1, num_episodes, alpha_eff, estimator.omega), end="")
        sys.stdout.flush()
    print("\nImplicit TD SARSA training finished.")
    return stats, np.array(val1_list), np.array(val2_list)

# ---------------------------
# EpisodeStats Class for Tracking Performance
# ---------------------------
class EpisodeStats():
    def __init__(self, episode_omega, episode_rewards):
        self.episode_omega=episode_omega
        self.episode_rewards = episode_rewards

# ---------------------------
# Helper Function to Run One Experiment
# ---------------------------
def run_experiment(algorithm, alpha0, num_episodes, c_alpha=1.0, epsilon=0.1, epsilon_decay=1.0, sanity_every=1, R=None):
    """
    Runs one experiment with a fresh estimator using the given algorithm.
    Returns:
       omegaeward : mean of average reward 
       ins_reward : the last reported instant reward
    """
    ## Initalize 
    estimator = Estimator(num_actions=num_actions, action_set=action_set)
    stats, val1_curve, val2_curve = algorithm(env, estimator, num_episodes, c_alpha, epsilon, epsilon_decay, alpha0, sanity_every, R=R)
    omegaeward = stats.episode_omega
    return val1_curve, omegaeward, val2_curve

# ---------------------------
# Run Multiple Experiments and Average Results
# ---------------------------
def multi_experiment(algorithm, alpha0, num_episodes, num_experiments, **kwargs):
    sanity_all = [] # used for the code sanity check
    omega_all = []
    val2_all = []
    for i in range(num_experiments):
        print("\nExperiment {}/{}".format(i+1, num_experiments))
        sanity_check, omegaeward, val2_curve = run_experiment(algorithm, alpha0, num_episodes, **kwargs)
        sanity_all.append(sanity_check)
        omega_all.append(omegaeward)
        val2_all.append(val2_curve)
    sanity_all = np.array(sanity_all)      # shape: (num_experiments, num_episodes)
    omega_all = np.array(omega_all)  # shape: (num_experiments, num_episodes)
    val2_all = np.array(val2_all)

    return sanity_all, omega_all, val2_all  


def arguments_generator():
    parser = argparse.ArgumentParser(description="For experiments")
    
    # Define the expected arguments
    parser.add_argument('--env', type=str, choices=['access_control','pendulum', 'cartpole'], required=True,
                        help='Name of the environment to run.')
    parser.add_argument('--name', type=str, default="default",
                        help='Prefix for the results figure filename.')
    parser.add_argument('--num_episodes', type=int, default=500,
                        help='Prefix for the results figure filename.')
    parser.add_argument('--num_experiments', type=int, default=1,
                        help='Prefix for the results figure filename.')
    parser.add_argument('--c_alpha','--c', type=float, default=1.0,
                    help='c_alpha, the value for the ratio')
    parser.add_argument('--lamb','--l', type=float, default=0.25,
                help='Lambda value for the TD(lamb)') 
    parser.add_argument(
        "--step_size_schedule",
        type=str,
        choices=["constant", "non_linear_decay", "s_decay"],
        help="Choose the step size schedule to use.")
    parser.add_argument(
        "--s",
        type=float,
        default=1.0)               
    args = parser.parse_args()

    return args 
# ---------------------------
# Main Execution: Run Experiments for Both Methods and Parameter Settings
# ---------------------------
if __name__ == '__main__':
    np.random.seed(20252026)

    # Define arguments 
    args = arguments_generator()
    if not (0.5 <= args.s <= 1.0):
        raise ValueError("For 's_decay', s must be in the range [0.5, 1.0].")

    # ---------------------------
    # Environment and Feature Preprocessing
    # ---------------------------

    env_name = args.env
    if env_name == "cartpole":
        # cartpole example 
        env = gym.make('CartPole-v1')
        raw_state_dim = env.observation_space.shape[0]  
        num_actions = env.action_space.n
        action_set = [i for i in range(num_actions)] 
        env = InfiniteHorizonWrapper(env)

    elif env_name == "pendulum":
        # pendulum example 
        if not hasattr(np, 'float_'):
            np.float_ = np.float64
        # use Gynmasium example 
        env = gym.make("Pendulum-v1")
        env_test = gym.make("Pendulum-v1") # dummy enviroment used for sanity check 
        # discretize action sets 
        action_set = np.linspace(-2, 2, 5).reshape(-1,1)
        num_actions = len(action_set)

        # scale reward function, and remove the termination stae 
        env = CustomInfinitePendulumWrapper(env)

        # make feature function 
        env_test = CustomInfinitePendulumWrapper(env_test)
        states = [env.observation_space.sample() for _ in range(10000)]
        featurizer = Pipeline([
            ("to_array", FunctionTransformer(lambda X: np.vstack(X), validate=False)),
            ("rbfs",    FeatureUnion([
                ("rbf3", RBFSampler(gamma=1.0, n_components=150)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=150))
            ])),
        ])
        featurizer.fit(states)

        # initial step sizes 
        alphas = np.arange(0.5, 3.05, 0.5) * 200  
        epsilon = 0.25

    elif env_name == "access_control":
        # make an access-control environment
        env = AccessControlEnv()
        env_test = AccessControlEnv() # dummy enviroment used for sanity check     
        
        num_actions = env.action_space.n
        action_set = [i for i in range(num_actions)] 
        states = [env.observation_space.sample() for _ in range(10000)]

        # make feature function 
        featurizer = Pipeline([
            ("to_array", FunctionTransformer(lambda X: np.array(X, dtype=float), validate=False)),
            ("scaler",  MinMaxScaler(feature_range=(0,1))),
            ("rbf",     RBFSampler(gamma=1.0, n_components=20, random_state=42))
        ])

        featurizer.fit(states)

        # initial step sizes
        alphas = np.arange(0.5, 3.05, 0.5) * 200 
        epsilon = 0.25

    num_experiments = args.num_experiments
    num_episodes = args.num_episodes + 1
    Lambda = args.lamb
    c_alpha = args.c_alpha


    # evaluation for sanity check 
    sanity_every = 1500
    # defines projection radius 
    radius_1 = 1000
    radius_2 = 5000

    env_setup = {
    "env_name": env_name,  # String representing the environment name.
    "env": env,  # The actual environment object.
    "action_set": action_set
    }  

    alpha2 = 400
    env_name = args.env 
    figure_name = args.name 
    num_alphas = len(alphas)
    num_episodes = args.num_episodes # number of iterations, however use term episodes in this code example 
    max_steps = 100  
    
    results = {}

    if args.step_size_schedule == "non_linear_decay":
        base_folder = os.path.join("result", "control_learning", env_name, "non_linear_decay")
    elif args.step_size_schedule == "s_decay":
        base_folder = os.path.join("result", "control_learning", env_name, "s_decay")
    else:
        raise ValueError

    # ---------------------------
    # Run Experiments for All Methods
    # ---------------------------

    for alpha0 in alphas:
        if args.step_size_schedule == "linear_decay":
            t = np.arange(1, num_episodes+1) +alpha2
            step_sizes = alpha0 / t
        elif args.step_size_schedule == "s_decay":
            print(args.step_size_schedule)
            step_sizes = np.full(num_episodes, alpha0 / alpha2)
            t = (np.arange(1, num_episodes+1) + alpha2) ** args.s # \beta_t = (initial_step size) / (t + 400)^{0.99}
            # first 150 iteration is fixed and set to be constant
            thr = 150 
            step_sizes[thr:] = alpha0 / t[:-thr] 
        else:
            # constant step size
            step_sizes = np.full(num_episodes, alpha0)
            t = np.arange(1, num_episodes+1)  +alpha2
            # first 150 iteration is fixed and set to be constant
            thr = 150 
            step_sizes[thr:] = alpha0 / t[:-thr]
        print("\n=== Running Implicit TD SARSA Experiments ===")

        print("\nImplicit TD SARSA with alpha0 =", alpha0)

        # implicit TD (lambda) method without projection
        sanity_all, omega_all, rewards_all = multi_experiment(
            sarsa_implicit, alpha0, num_episodes, num_experiments, c_alpha=c_alpha, epsilon=epsilon,
            sanity_every=sanity_every, R = None)
        results[("implicit", alpha0)] = (sanity_all, omega_all, rewards_all)
        # implicit TD (lambda) method with projection (radius 1) 
        sanity_all, omega_all, rewards_all = multi_experiment(
            sarsa_implicit, alpha0, num_episodes, num_experiments,
            c_alpha=c_alpha, epsilon=epsilon,
            sanity_every=sanity_every, R = radius_1)
        results[("implicit_proj_R1", alpha0)] = (sanity_all, omega_all, rewards_all)   
        # implicit TD (lambda) method with projection (radius 2) 
        sanity_all, omega_all, rewards_all = multi_experiment(
            sarsa_implicit, alpha0, num_episodes, num_experiments,
            c_alpha=c_alpha, epsilon=epsilon,
            sanity_every=sanity_every, R = radius_2)
        results[("implicit_proj_R2", alpha0)] = (sanity_all, omega_all, rewards_all)   
        
        print("\nStandard TD SARSA with alpha0 =", alpha0)
        
        # Standard TD (lambda) method
        sanity_all, omega_all, rewards_all = multi_experiment(
            sarsa_vanilla, alpha0, num_episodes, num_experiments,
            c_alpha=c_alpha, epsilon=epsilon,
            sanity_every=sanity_every)
        results[("standard", alpha0)] = (sanity_all, omega_all, rewards_all)

    # Store results 
    results[("alpha_2", 0)] = alpha2 
    results[("alphas",0)] = alphas

    os.makedirs(base_folder, exist_ok=True)
    folder_name = f"lam_{Lambda}_cal_{c_alpha}_c2_{alpha2}"
    folder_path = os.path.join(base_folder, folder_name)
    save_path = os.path.join(folder_path, "results_dict.pkl")

    os.makedirs(folder_path, exist_ok=True)  # Create the hyperparameter-specific folder if needed

    # Pickle the files
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results have been saved to: {save_path}")
