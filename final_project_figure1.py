#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:17:48 2024

@author: habbas
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the number of arms and rounds
num_arms = 5
num_rounds = 1000

# Simulate probabilities for arms using a Beta distribution
true_probabilities = np.random.beta(a=2, b=5, size=num_arms)

# Function to simulate pulling an arm
def pull_arm(arm, probabilities):
    return np.random.rand() < probabilities[arm]

# Define algorithms for comparison
def random_policy(probabilities, rounds):
    regrets = []
    cumulative_regret = 0
    for t in range(rounds):
        arm = np.random.choice(len(probabilities))
        reward = pull_arm(arm, probabilities)
        optimal_reward = max(probabilities)
        regret = optimal_reward - probabilities[arm]
        cumulative_regret += regret
        regrets.append(cumulative_regret)
    return regrets

def greedy_policy(probabilities, rounds):
    regrets = []
    cumulative_regret = 0
    arm_counts = np.zeros(len(probabilities))
    arm_rewards = np.zeros(len(probabilities))
    for t in range(rounds):
        if t < len(probabilities):  # Explore each arm once
            arm = t
        else:  # Exploit the best-known arm
            arm = np.argmax(arm_rewards / (arm_counts + 1e-6))
        reward = pull_arm(arm, probabilities)
        optimal_reward = max(probabilities)
        regret = optimal_reward - probabilities[arm]
        cumulative_regret += regret
        regrets.append(cumulative_regret)
        arm_counts[arm] += 1
        arm_rewards[arm] += reward
    return regrets

def ucb_policy(probabilities, rounds):
    regrets = []
    cumulative_regret = 0
    arm_counts = np.zeros(len(probabilities))
    arm_rewards = np.zeros(len(probabilities))
    for t in range(rounds):
        if t < len(probabilities):  # Explore each arm once
            arm = t
        else:  # Use UCB to select an arm
            confidence_bounds = arm_rewards / (arm_counts + 1e-6) + np.sqrt(
                2 * np.log(t + 1) / (arm_counts + 1e-6)
            )
            arm = np.argmax(confidence_bounds)
        reward = pull_arm(arm, probabilities)
        optimal_reward = max(probabilities)
        regret = optimal_reward - probabilities[arm]
        cumulative_regret += regret
        regrets.append(cumulative_regret)
        arm_counts[arm] += 1
        arm_rewards[arm] += reward
    return regrets

# Run simulations for each policy
random_regrets = random_policy(true_probabilities, num_rounds)
greedy_regrets = greedy_policy(true_probabilities, num_rounds)
ucb_regrets = ucb_policy(true_probabilities, num_rounds)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(random_regrets, label="Random Policy")
plt.plot(greedy_regrets, label="Greedy Policy")
plt.plot(ucb_regrets, label="UCB Policy")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Regret")
plt.title("Comparison of Bandit Policies (Cumulative Regret)")
plt.legend()
plt.grid()
plt.show()
