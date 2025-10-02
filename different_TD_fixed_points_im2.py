#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math 
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import sem
import argparse
import os 

from randomMRP import GenerateRandomMRP
from feature_matrix import generate_feature_matrix
from stationary_distribution import stationary_distribution
from gain_and_bias import stationary_reward, basic_bias
from theta_star import find_theta_star
from environment import BoyanChain



def linear_TD_lambda(mrp, Phi, theta_star, theta_e, Lambda, T, c_alpha, step_sizes, initial_theta):
    """
        Inputs:
                (1) mrp: markov reward process
                (2) Phi: feature matrix
                (3) theta_star: TD fixed point
                (4) theta_e: solution of Phi theta = e
                (5) Lambda: algorithm lambda parameter
                (6) T: number of iterations
                (7) c_alpha: algorithm step size parameter
                (8) step_sizes: step-size sequence
                (9) initial_theta: theta_0

        Output:
                (1) proj_diff_norm_hist:
                (2) proj_diff_hist:

    """

    d = Phi.shape[1]

    # Initialization
    bar_r = 0.0
    theta = np.copy(initial_theta)
    z = np.zeros(d)
    proj_diff_norm_hist = np.zeros(T)
    proj_diff_hist = np.zeros(T)
    mrp.reset_initial_state()

    for t in range(T):
        # Observe data
        current_state = mrp.current_state
        r, next_state = mrp.step()

        # Get TD error
        delta = r - bar_r + np.dot(Phi[next_state, :], theta) - np.dot(Phi[current_state, :], theta)

        # Update eligibility trace
        z = Lambda * z + Phi[current_state, :]

        # Update average-reward estimate
        bar_r = bar_r + c_alpha * step_sizes[t] * (r - bar_r)

        # Update parameter vector
        theta = theta + step_sizes[t] * delta * z
        

        # Norm of projection of theta_t - theta^* onto E
        projected_theta = theta - (np.dot(theta, theta_e) / np.dot(theta_e, theta_e)) * theta_e
        proj_diff_norm = LA.norm(projected_theta - theta_star) **2 +  (bar_r - gain)**2 
        proj_diff_norm_hist[t] = proj_diff_norm

        # projection of theta_t - theta^* onto theta_e
        proj_diff_hist[t] = np.dot(theta, theta_e) / LA.norm(theta_e)

    return proj_diff_norm_hist, proj_diff_hist

def two_imp_linear_TD_lambda(mrp, Phi, theta_star, theta_e, Lambda, T, c_alpha, step_sizes, initial_theta, radius = 100):
    """
        Inputs:
                (1) mrp: markov reward process
                (2) Phi: feature matrix
                (3) theta_star: TD fixed point
                (4) theta_e: solution of Phi theta = e
                (5) Lambda: algorithm lambda parameter
                (6) T: number of iterations
                (7) c_alpha: algorithm step size parameter
                (8) step_sizes: step-size sequence
                (9) initial_theta: theta_0

        Output:
                (1) proj_diff_norm_hist:
                (2) proj_diff_hist:

    """

    d = Phi.shape[1]

    # Initialization
    bar_r = 0.0
    theta = np.copy(initial_theta)
    z = np.zeros(d)
    proj_diff_norm_hist = np.zeros(T)
    proj_diff_hist = np.zeros(T)
    mrp.reset_initial_state()

    for t in range(T):
        # Observe data
        current_state = mrp.current_state
        r, next_state = mrp.step()

        # Get TD error
        delta = r - bar_r + np.dot(Phi[next_state, :], theta) - np.dot(Phi[current_state, :], theta)

        # Update eligibility trace
        z = Lambda * z + Phi[current_state, :]

        # Update average-reward estimate
        bar_r = bar_r + c_alpha * step_sizes[t] * (r - bar_r) / (1 + c_alpha * step_sizes[t])

        # Update parameter vector
        theta = theta + step_sizes[t] * delta * z / (1 + step_sizes[t]*np.linalg.norm(z)**2)

        if radius:
            if np.linalg.norm(theta) > radius - 1:
                theta = theta / np.linalg.norm(theta) * (radius - 1)
            if np.linalg.norm(bar_r) > 1:
                bar_r = bar_r / np.linalg.norm(bar_r) * 1.0

        # Norm of projection of theta_t - theta^* onto E
        projected_theta = theta - (np.dot(theta, theta_e) / np.dot(theta_e, theta_e)) * theta_e
        proj_diff_norm = LA.norm(projected_theta - theta_star) **2 +  (bar_r - gain)**2 
        proj_diff_norm_hist[t] = proj_diff_norm
        
        # projection of theta_t - theta^* onto theta_e
        proj_diff_hist[t] = np.dot(theta-theta_star, theta_e) / LA.norm(theta_e)
    #print(bar_r, gain)
    
    return proj_diff_norm_hist, proj_diff_hist


if __name__ == "__main__":
    np.random.seed(20252026)
    # arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--env",type=str,default="MRP"
    )
    parser.add_argument(
        "-l", "--lamb",type=float, default=0.25
    )
    parser.add_argument(
        "-c", "--c_alpha",type=float, default=1.0
    )
    parser.add_argument(
        "--step_size_schedule",
        type=str,
        choices=["constant", "non_linear_decay", "s_decay"],
        help="Choose the step size schedule to use."
    )
    parser.add_argument(
        "--s",
        type=float,
        default=1.0
    )

    args = parser.parse_args()
    num_states = 100

    if not (0.5 <= args.s <= 1.0):
        raise ValueError("For 's_decay', s must be in the range [0.5, 1.0].")

    if args.env=="MRP":
        d = 10
    elif args.env =="Boyan":
        d = 4 + 2# fixed for Boyan
        num_states = 100

    if args.env == "MRP": 
        mrp = GenerateRandomMRP(num_states=num_states)
        pi = stationary_distribution(mrp.trans_prob) # computing stationary distribution 
        gain = stationary_reward(mrp.rewards, pi) 
        bias = basic_bias(mrp.rewards, mrp.trans_prob, gain, pi) 
        # Generating Feature Matrix and Figuring out the Answer
        Phi = generate_feature_matrix(mrp.num_states, d, bias)
        theta_star = find_theta_star(Phi, bias)
        theta_e = LA.lstsq(Phi, np.ones(mrp.num_states), rcond=None)[0]
        
    # Setting the Hyperparameters & Experiment conditions 
    Lambda = args.lamb
    T = 5000
    c_alpha = args.c_alpha
    num_exp = 50
    #linear_decay = args.step_size_schedule
    radius_1 = 1000
    radius_2 = 5000
    if args.step_size_schedule == "non_linear_decay":
        alphas = np.arange(0.05, 5.05, 0.05) * 1000 # for MRP
    elif args.step_size_schedule == "constant":
        alphas = np.arange(0.05, 3.05, 0.05) * 1000 
    elif args.step_size_schedule == "s_decay":
        alphas = np.arange(0.05, 5.05, 0.05) * 1000 
    alpha2 = 500
    num_alphas = len(alphas)


    # Store Experiment results 
    proj_diff_exp = np.zeros((num_exp, num_alphas, T))
    proj_diff_exp_im = np.zeros((num_exp, num_alphas, T))
    proj_2_diff_exp_im = np.zeros((num_exp, num_alphas, T))
    proj_3_diff_exp_im = np.zeros((num_exp, num_alphas, T))

    proj_diff_norm_exp = np.zeros((num_exp, num_alphas, T))
    proj_diff_norm_exp_im = np.zeros((num_exp, num_alphas, T))
    proj_2_diff_norm_exp_im = np.zeros((num_exp, num_alphas, T))
    proj_3_diff_norm_exp_im = np.zeros((num_exp, num_alphas, T))
    print(f"Current Setting: LD_{args.step_size_schedule}_d_{d}_num_{num_states}_lam_{Lambda}_cal_{c_alpha}_c2_{alpha2}")
    
    for i in range(num_exp):
        print(f"initial point No. {i}")
        if args.env == "Boyan":
            policy_action = np.random.binomial(n=1, p=0.5, size=13)
            # for Boyan, we change the policies, following the binomial distribution  
            mrp = BoyanChain(eval_bool=True, policy_action=policy_action)
            pi = stationary_distribution(mrp.trans_prob)
            gain = stationary_reward(mrp.rewards, pi)
            bias = basic_bias(mrp.rewards, mrp.trans_prob, gain, pi) # Basic differential 
            #Phi = generate_feature_matrix(mrp.num_states, d, bias)
            Phi = mrp.build_feature_matrix(bias=bias)
            theta_star = find_theta_star(Phi, bias)
            
            theta_e = LA.lstsq(Phi, np.ones(mrp.num_states), rcond=None)[0]
            #print(theta_star, gain, theta_e)
        initial_theta = np.random.uniform(-1,1,d)  

        exp = i
        for alpha_idx, alpha in enumerate(alphas):
            # Setup the step schedules 
            if args.step_size_schedule == "constant":
                step_sizes = np.full(T, alpha / alpha2)
                #t = np.arange(1, T+1) +alpha2
                #step_sizes = alpha / t
            elif args.step_size_schedule == "non_linear_decay":
                step_sizes = np.full(T, alpha / alpha2)
                t = np.arange(1, T+1) +alpha2
                thr = 150 
                step_sizes[thr:] = alpha / t[:-thr]
            elif args.step_size_schedule == "s_decay":
                initial_step_size = alpha / alpha2 
                step_sizes = np.full(T, initial_step_size)
                t = np.arange(1, T+1) ** args.s
                thr = 150 
                step_sizes[thr:] = initial_step_size/ t[:-thr]

            # Standard algorithm 
            proj_diff_norm_hist_regular, proj_diff_hist_regular = linear_TD_lambda(mrp, Phi, theta_star, theta_e, Lambda, T, c_alpha, step_sizes, initial_theta)
            
            # implicit update without Projection on Radius 
            proj_diff_norm_hist_two_im, proj_diff_hist_two_im = two_imp_linear_TD_lambda(mrp, Phi, theta_star, theta_e, Lambda, T, c_alpha, step_sizes, initial_theta, radius=None)
            # With Projection on Radius 
            proj_2_diff_norm_hist_two_im, proj_2_diff_hist_two_im = two_imp_linear_TD_lambda(mrp, Phi, theta_star, theta_e, Lambda, T, c_alpha, step_sizes, initial_theta, radius=radius_1)
            proj_3_diff_norm_hist_two_im, proj_3_diff_hist_two_im = two_imp_linear_TD_lambda(mrp, Phi, theta_star, theta_e, Lambda, T, c_alpha, step_sizes, initial_theta, radius=radius_2)
            
            proj_diff_exp[exp, alpha_idx, :] = proj_diff_hist_regular
            proj_diff_exp_im[exp, alpha_idx, :] = proj_diff_hist_two_im
            proj_2_diff_exp_im[exp, alpha_idx, :] = proj_2_diff_hist_two_im
            proj_3_diff_exp_im[exp, alpha_idx, :] = proj_3_diff_hist_two_im

            proj_diff_norm_exp[exp, alpha_idx, :] = proj_diff_norm_hist_regular
            proj_diff_norm_exp_im[exp, alpha_idx, :] = proj_diff_norm_hist_two_im
            proj_2_diff_norm_exp_im[exp, alpha_idx, :] = proj_2_diff_norm_hist_two_im
            proj_3_diff_norm_exp_im[exp, alpha_idx, :] = proj_3_diff_norm_hist_two_im

    method_data = [
        proj_diff_norm_exp,
        proj_diff_norm_exp_im,
        proj_2_diff_norm_exp_im,
        proj_3_diff_norm_exp_im
    ]

    # Corresponding method names (for documentation purposes)
    method_names = ["Regular", "Implicit", "Proj-Implicit (R_1)", "Proj-Implicit (R_2)"]

    # Choose the base folder depending on the linear_decay flag.
    base_folder = os.path.join("result", "evaluation", args.env, args.step_size_schedule)

    # Ensure the base folder exists
    os.makedirs(base_folder, exist_ok=True)

    # Define the folder name based on the hyperparameters
    folder_name = f"d_{d}_num_{num_states}_lam_{Lambda}_cal_{c_alpha}_c2_{alpha2}"
    folder_path = os.path.join(base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)  # Create the hyperparameter-specific folder if needed

    # Pack all results into a dictionary for convenient saving using np.savez
    results = {
        "theta_star": theta_star,
        "theta_e": theta_e,
        "proj_diff_norm_exp": proj_diff_norm_exp,
        "proj_diff_norm_exp_im": proj_diff_norm_exp_im,
        "proj_2_diff_norm_exp_im": proj_2_diff_norm_exp_im,
        "proj_3_diff_norm_exp_im": proj_3_diff_norm_exp_im,
        "method_names": method_names,
        "alphas": alphas
    }

    # Construct the output file path
    save_path = os.path.join(folder_path, "results.npz")

    # Save the results using numpy's savez; this creates a compressed archive file.
    np.savez(save_path, **results)

    print(f"Results have been saved to: {save_path}")