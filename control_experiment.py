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

# Wrapper 
class CustomInfiniteAcrobotWrapper(gym.Wrapper):
    """
    A wrapper for the Acrobot-v1 environment that:
      1. Modifies the reward to be based on the tip height.
      2. Forces an infinite-horizon formulation (i.e. no terminal state).
    
    The tip height is computed as:
         tip_height = -cos(theta1) - cos(theta1 + theta2)
    where theta1 and theta2 are recovered from the first four entries of the
    observation (assumed to be [cos(theta1), sin(theta1), cos(theta2), sin(theta2), ...]).
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        # Take a step in the base environment.
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Retrieve the first four components: [cos(theta1), sin(theta1), cos(theta2), sin(theta2)]
        cos_t1, sin_t1, cos_t2, sin_t2 = observation[:4]
        # Recover the angles.
        theta1 = np.arctan2(sin_t1, cos_t1)
        theta2 = np.arctan2(sin_t2, cos_t2)
        # Compute the tip height.
        tip_height = -np.cos(theta1) - np.cos(theta1 + theta2)
        # Use tip_height as the new reward. Optionally, you can rescale or shift this value.
        new_reward = tip_height
        truncated = False
        if tip_height > 1.0:
            print("succeed")
            new_reward = 1.0#100.0
            truncated = True
        else : 
            new_reward = (new_reward + 2.01) / 10
        # Force the episode not to end by setting terminated and truncated to False.
        return observation, new_reward, True, info
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

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
        new_reward = new_reward / 16.27#(new_reward + 16.27) / 16.27
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
        #print(example_state)
        self.num_features = len(self.featurize_state(example_state))
        # Initialize one weight vector per action.
        self.weights = [np.random.uniform(-0.5, 0.5, size=self.num_features)for _ in range(self.nA)] #np.zeros(self.num_features) 
        self.bar_r = 0.0 
    
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
        A[best_action_idx] += (1.0 - epsilon)
        return A
    return policy_fn

# ---------------------------
# RMSBE Computation
# ---------------------------
def compute_rmsbe(estimator, env, discount_factor=1.0, n_samples=100):
    """
    Computes the True estimation error of the Q function and reward estimation 
    """
    errors = []
    errors_reward = []
    length = 50
    for j in range(10):
        s = env_test.reset() 
        for _ in range(10):
            a_idx = np.argmax(estimator.predict(s))
            a = estimator.action_set[a_idx]
            s, _, _, _ = env_test.step(a)
        td_error = 0.0 
        reward_est = 0.0 
        a_idx = np.argmax(estimator.predict(s))
        a = estimator.action_set[a_idx]
        q_sa = estimator.predict(s, a_idx)

        for _ in range(n_samples):
            # estimate 
            a_idx = np.argmax(estimator.predict(s))
            a = estimator.action_set[a_idx]
            new_state, reward, terminated, _ = env_test.step(a)
            reward_est += reward 
            td_error += reward - estimator.bar_r 
            q_sa = estimator.predict(s, a_idx)
            a_next_idx = np.argmax(estimator.predict(new_state))
            
            q_next = estimator.predict(new_state, a_next_idx)
            s = new_state 

        bar_r_est_error = reward_est/n_samples - estimator.bar_r
        q_error = td_error - q_sa 
        errors.append(q_error**2)
        errors_reward.append(bar_r_est_error**2)
    
    return np.sqrt(np.mean(errors)), np.sqrt(np.mean(errors_reward))

# ---------------------------
# SARSA with Vanilla TD Update
# ---------------------------
def sarsa_vanilla(env_dict, estimator, num_episodes, c_alpha=1.0,
                  epsilon=0.1, epsilon_decay=1.0, alpha0=0.1, rmsbe_every=1, n_rmsbe_samples=100, R = 1000, max_step=100):
    """
    Implements SARSA using the vanilla TD update under AR
    Also computes RMSBE at the end of each episode.
    Returns stats and an array of RMSBE values.
    """
    
    nA = estimator.nA#env.action_space.n
    stats = EpisodeStats(episode_bar_r=np.zeros(num_episodes),
                         episode_rewards=np.zeros(num_episodes))
    rmsbe_list = []
    reward_list = []
    state = env.reset()
    z = 0.0

    for i_episode in range(num_episodes):
        #if (i_episode) % 250 == 0:
        #    state = env.reset()
        alpha_eff=step_sizes[i_episode]
        if i_episode < thr:
            current_epsilon = 0.25
        elif i_episode < thr*2:
            current_epsilon = 0.1
        else:
            current_epsilon = 0.00
        alpha_eff=step_sizes[i_episode]
        current_epsilon = epsilon * (epsilon_decay ** i_episode)
        policy = make_epsilon_greedy_policy(estimator, current_epsilon, nA)
        action_idx = np.random.choice(nA, p=policy(state))
        action = estimator.action_set[action_idx]
        #for t in itertools.count():
        new_state, reward, terminated, _ = env.step(action)
        new_action_idx = np.random.choice(nA, p=policy(new_state))
        new_action = estimator.action_set[new_action_idx]

        stats.episode_rewards[i_episode] += reward
        stats.episode_bar_r[i_episode] += estimator.bar_r
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
        td_target = reward - estimator.bar_r + q_next
        td_error = td_target - q_sa
        z = Lambda * z + phi_state
        estimator.weights[action_idx] += (alpha_eff * td_error * z) 
        estimator.bar_r = estimator.bar_r + c_alpha * alpha_eff* (reward - estimator.bar_r) 
        state = new_state

        if (i_episode) % rmsbe_every == 0:
            rmsbe, reward_curve = compute_rmsbe(estimator, env, c_alpha, n_samples=n_rmsbe_samples)
            rmsbe_list.append(rmsbe)
            reward_list.append(reward_curve)
        print("\rStandard TD SARSA Episode {}/{} - RMEST: {:.3g} - α_eff: {:.4f} - AR_{:.4g}".format(
            i_episode+1, num_episodes, rmsbe, alpha_eff, estimator.bar_r), end="")
        sys.stdout.flush()
    print("\nStandard TD SARSA training finished.")
    return stats, np.array(rmsbe_list), np.array(reward_list)

# ---------------------------s
# SARSA with Implicit TD Update
# ---------------------------
def sarsa_implicit(env_dict, estimator, num_episodes, c_alpha=1.0,
                   epsilon=0.1, epsilon_decay=1.0, alpha0=0.5, rmsbe_every=1, n_rmsbe_samples=100, R = 1000, max_step=100):
    """
    Implements SARSA with an implicit TD update under AR 
    Also computes RMSBE at the end of each episode.
    Returns stats and an array of RMSBE values.
    """
    nA = estimator.nA#env.action_space.n
    stats = EpisodeStats(episode_bar_r=np.zeros(num_episodes),
                         episode_rewards=np.zeros(num_episodes))
    rmsbe_list = []
    reward_list = []
    z = 0.0 
    state = env.reset()
    thr = 5000 
    current_epsilon =0.25
    policy = make_epsilon_greedy_policy(estimator, current_epsilon, nA)
    action_idx = np.random.choice(nA, p=policy(state))
    
    for i_episode in range(num_episodes):

        alpha_eff=step_sizes[i_episode]
        if i_episode < thr:
            current_epsilon = 0.25
        elif i_episode < thr*2:
            current_epsilon = 0.1
        else:
            current_epsilon = 0.00#epsilon * (epsilon_decay ** (i_episode-thr))
        policy = make_epsilon_greedy_policy(estimator, current_epsilon, nA)
        action = estimator.action_set[action_idx]
        new_state, reward, terminated, _ = env.step(action)
        new_action_idx = np.random.choice(nA, p=policy(new_state))
        new_action = estimator.action_set[new_action_idx]
        stats.episode_rewards[i_episode] += reward
        stats.episode_bar_r[i_episode] += estimator.bar_r
        phi_state = estimator.featurize_state(state)
        q_sa = np.dot(estimator.weights[action_idx], phi_state)

        if terminated:
            new_state = env.reset()
            phi_next = estimator.featurize_state(new_state)
            q_next = np.dot(estimator.weights[new_action_idx], phi_next)
        else:
            phi_next = estimator.featurize_state(new_state)
            q_next = np.dot(estimator.weights[new_action_idx], phi_next)
        td_target = reward - estimator.bar_r + q_next
        td_error = td_target - q_sa
        z = Lambda * z + phi_state
        estimator.weights[action_idx] += (alpha_eff * td_error * z) / (1 + alpha_eff * np.linalg.norm(phi_state)**2)
        estimator.bar_r = estimator.bar_r + c_alpha * alpha_eff* (reward - estimator.bar_r) / (1+ c_alpha * alpha_eff)

        if R:
            if np.linalg.norm(estimator.weights[action_idx]) > R - 1:
                estimator.weights[action_idx] = (
                    estimator.weights[action_idx] 
                    / np.linalg.norm(estimator.weights[action_idx]) 
                    * (R - 1)
                )
            if np.linalg.norm(estimator.bar_r) > 1:
                estimator.bar_r = (
                    estimator.bar_r 
                    / np.linalg.norm(estimator.bar_r) 
                    * 1.0
                )

        state = new_state
        action_idx = new_action_idx

        if (i_episode) % rmsbe_every == 0:
            rmsbe, reward_curve = compute_rmsbe(estimator, env, c_alpha, n_samples=n_rmsbe_samples)
            rmsbe_list.append(rmsbe)
            reward_list.append(reward_curve)
        print("\rImplicit TD SARSA Episode {}/{} - RMEST: {} - α_eff: {:.5f} - AR_{:.4f}".format(
            i_episode+1, num_episodes, rmsbe, alpha_eff, estimator.bar_r), end="")
        sys.stdout.flush()
    print("\nImplicit TD SARSA training finished.")
    return stats, np.array(rmsbe_list), np.array(reward_list)

# ---------------------------
# EpisodeStats Class for Tracking Performance
# ---------------------------
class EpisodeStats():
    def __init__(self, episode_bar_r, episode_rewards):
        self.episode_bar_r=episode_bar_r
        self.episode_rewards = episode_rewards

# ---------------------------
# Helper Function to Run One Experiment
# ---------------------------
def run_experiment(algorithm, alpha0, num_episodes, c_alpha=1.0, epsilon=0.1, epsilon_decay=1.0, rmsbe_every=1, n_rmsbe_samples=100, R=None):
    """
    Runs one experiment with a fresh estimator using the given algorithm.
    Returns:
       rmsbe_curve: Array of RMSBE values per episode.
       cum_reward: Array of cumulative reward per episode.
       episode_rewards: Array of episode rewards.
    """
    ## Initalize 
    estimator = Estimator(num_actions=num_actions, action_set=action_set)
    stats, rmsbe_curve, reward_curve = algorithm(env, estimator, num_episodes, c_alpha, epsilon, epsilon_decay, alpha0, rmsbe_every, n_rmsbe_samples, R=R)

    #cum_reward = np.cumsum(stats.episode_rewards)
    bar_reward = stats.episode_bar_r
    ins_reward = reward_curve #stats.episode_rewards
    return rmsbe_curve, bar_reward, ins_reward#stats.episode_rewards

# ---------------------------
# Run Multiple Experiments and Average Results
# ---------------------------
def multi_experiment(algorithm, alpha0, num_episodes, num_experiments, **kwargs):
    rmsbe_all = []
    bar_reward_all = []
    rewards_all = []
    for i in range(num_experiments):
        print("\nExperiment {}/{}".format(i+1, num_experiments))
        rmsbe_curve, bar_reward, ins_reward = run_experiment(algorithm, alpha0, num_episodes, **kwargs)
        rmsbe_all.append(rmsbe_curve)
        bar_reward_all.append(bar_reward)
        rewards_all.append(ins_reward)
    rmsbe_all = np.array(rmsbe_all)      # shape: (num_experiments, num_episodes)
    bar_reward_all = np.array(bar_reward_all)  # shape: (num_experiments, num_episodes)
    rewards_all = np.array(rewards_all)

    return rmsbe_all, bar_reward_all, rewards_all  

# ---------------------------
# Plotting Helpers
# ---------------------------
def plot_with_variance(x, mean, std, title, xlabel, ylabel):
    plt.figure(figsize=(8, 4))
    plt.plot(x, mean, label="Mean")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2, label="± 1 Std")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def arguments_generator():
    parser = argparse.ArgumentParser(description="For experiments")
    
    # Define the expected arguments
    parser.add_argument('--env', type=str, choices=['access_control','pendulum', 'cartpole', 'mountain_car', 'acrobot'], required=True,
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
    args = arguments_generator()
    if not (0.5 <= args.s <= 1.0):
        raise ValueError("For 's_decay', s must be in the range [0.5, 1.0].")

    # ---------------------------
    # Environment and Feature Preprocessing
    # ---------------------------

    env_name = args.env
    if env_name == "cartpole":
        env = gym.make('CartPole-v1')
        raw_state_dim = env.observation_space.shape[0]  
        num_actions = env.action_space.n
        action_set = [i for i in range(num_actions)] 
        env = InfiniteHorizonWrapper(env)
    elif env_name == "pendulum":
        if not hasattr(np, 'float_'):
            np.float_ = np.float64
        env = gym.make("Pendulum-v1")
        env_test = gym.make("Pendulum-v1")
        action_set = np.linspace(-2, 2, 5).reshape(-1,1)
        num_actions = len(action_set)

        env = CustomInfinitePendulumWrapper(env)
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
        alphas = np.arange(0.5, 3.05, 0.5) * 200  

        epsilon = 0.25
    elif env_name == "access_control":
        env = AccessControlEnv()
        env_test = AccessControlEnv()  # used to measure RMSBE    
        num_actions = env.action_space.n
        action_set = [i for i in range(num_actions)] 
        # 11 X 4 possible combinations arise in state space 
        states = [env.observation_space.sample() for _ in range(10000)]

        featurizer = Pipeline([
            ("to_array", FunctionTransformer(lambda X: np.array(X, dtype=float), validate=False)),
            ("scaler",  MinMaxScaler(feature_range=(0,1))),
            ("rbf",     RBFSampler(gamma=1.0, n_components=20, random_state=42))
        ])

        featurizer.fit(states)
        alphas = np.arange(0.5, 3.05, 0.5) * 200 
        epsilon = 0.25


    num_experiments = args.num_experiments
    num_episodes = args.num_episodes + 1
    Lambda = args.lamb
    c_alpha = args.c_alpha

    epsilon_decay = 1.0
    rmsbe_every = 1500
    n_rmsbe_samples = 100
    linear_decay = False 
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
    num_episodes = args.num_episodes  
    max_steps = 100  
    
    results = {}
    #print(args.step_size_schedule)
    if args.step_size_schedule == "non_linear_decay":
        base_folder = os.path.join("result", "control_learning", env_name, "non_linear_decay")
    elif args.step_size_schedule == "s_decay":
        base_folder = os.path.join("result", "control_learning", env_name, "s_decay")
    else:
        raise ValueError

    for alpha0 in alphas:
        if args.step_size_schedule == "linear_decay":
            t = np.arange(1, num_episodes+1) +alpha2
            step_sizes = alpha0 / t
        elif args.step_size_schedule == "s_decay":
            print(args.step_size_schedule)
            #step_sizes = np.full(num_episodes, initial_step_size)
            step_sizes = np.full(num_episodes, alpha0 / alpha2)
            t = (np.arange(1, num_episodes+1) + alpha2) ** args.s
            thr = 150 
            step_sizes[thr:] = alpha0 / t[:-thr]
        else:
            step_sizes = np.full(num_episodes, alpha0)
            t = np.arange(1, num_episodes+1)  +alpha2
            thr = 150 
            #step_sizes[:thr] = alpha / alpha2
            step_sizes[thr:] = alpha0 / t[:-thr]
        print("\n=== Running Implicit TD SARSA Experiments ===")
    #for alpha0 in implicit_params:

        print("\nImplicit TD SARSA with alpha0 =", alpha0)
        rmsbe_all, bar_reward_all, rewards_all = multi_experiment(
            sarsa_implicit, alpha0, num_episodes, num_experiments, c_alpha=c_alpha, epsilon=epsilon, epsilon_decay=epsilon_decay,
            rmsbe_every=rmsbe_every, n_rmsbe_samples=n_rmsbe_samples, R = None)
        results[("implicit", alpha0)] = (rmsbe_all, bar_reward_all, rewards_all)
        rmsbe_all, bar_reward_all, rewards_all = multi_experiment(
            sarsa_implicit, alpha0, num_episodes, num_experiments,
            c_alpha=c_alpha, epsilon=epsilon, epsilon_decay=epsilon_decay,
            rmsbe_every=rmsbe_every, n_rmsbe_samples=n_rmsbe_samples, R = radius_1)
        results[("implicit_proj_R1", alpha0)] = (rmsbe_all, bar_reward_all, rewards_all)   
        rmsbe_all, bar_reward_all, rewards_all = multi_experiment(
            sarsa_implicit, alpha0, num_episodes, num_experiments,
            c_alpha=c_alpha, epsilon=epsilon, epsilon_decay=epsilon_decay,
            rmsbe_every=rmsbe_every, n_rmsbe_samples=n_rmsbe_samples, R = radius_2)
        results[("implicit_proj_R2", alpha0)] = (rmsbe_all, bar_reward_all, rewards_all)   
        print("\nStandard TD SARSA with alpha0 =", alpha0)
        #alpha0 = alpha0 
        rmsbe_all, bar_reward_all, rewards_all = multi_experiment(
            sarsa_vanilla, alpha0, num_episodes, num_experiments,
            c_alpha=c_alpha, epsilon=epsilon, epsilon_decay=epsilon_decay,
            rmsbe_every=rmsbe_every, n_rmsbe_samples=n_rmsbe_samples)
        results[("standard", alpha0)] = (rmsbe_all, bar_reward_all, rewards_all)

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
    
    episodes = np.arange(1, num_episodes+1)
    print(f"Results have been saved to: {save_path}")
