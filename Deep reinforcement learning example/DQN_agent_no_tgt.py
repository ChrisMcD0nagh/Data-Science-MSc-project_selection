import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import keras
from keras.activations import relu, linear
from time import time
import sys

# Set up neural network structure as an MLP
learning_rate = 0.001
base_model = keras.Sequential()
base_model.add(keras.layers.Dense(64, input_dim=8, activation=relu))
base_model.add(keras.layers.Dense(64, activation=relu))
base_model.add(keras.layers.Dense(4, activation=linear)) # Assuming discrete action space (So 4 actions) - note predicted state-action values can be negative
base_model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=learning_rate )) # Consider Huber loss

def experience_replay():
    if len(memory) >= batch_size: # Ensures there are enough state-action values to train on
        sample_choices = np.array(memory, dtype='object')

        # Only ever train the NN based on n=batch_size number of samples
        mini_batch_index = np.random.choice(len(sample_choices), batch_size)
        states = []
        actions = []
        next_states = []
        rewards = []
        finishes = []

        # Add all the states/actions/next_states/rewards/terminal values to their own lists
        for index in mini_batch_index:
            states.append(memory[index][0])
            actions.append(memory[index][1])
            next_states.append(memory[index][2])
            rewards.append(memory[index][3])
            finishes.append(memory[index][4])
        states = np.concatenate(states, axis=0)
        actions = np.array(actions)
        next_states = np.concatenate(next_states, axis=0)
        rewards = np.array(rewards)
        finishes = np.array(finishes)
        
        # Predict q_vals of subsequent states
        global target_net
        q_vals_next_state = model.predict_on_batch(next_states)
        
        # Predict q_vals of preceeding states  
        q_vals_target = model.predict_on_batch(states)

        # Update q_vals of preceeding states with the predicted (max) q_vals of subsequent states
        max_q_values_next_state = np.amax(q_vals_next_state, axis=1)
        q_vals_target[np.arange(batch_size), actions] = rewards + gamma * (max_q_values_next_state) * (1 - finishes)
        
        # Retrain the NN each iteration
        model.fit(states, q_vals_target, verbose=0, epochs=1)
        
        global epsilon
        if epsilon > min_eps:  # Linear epsilon decay with a minimum threshold
            epsilon *= 0.996

memory = deque(maxlen=1000000)
batch_size = 64
epsilon=1
min_eps=0.01 
gamma=0.99

model = base_model

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

def play(num_episodes = 1, checkp = 50):
    
    scores  = []
    for i in range(1, num_episodes+1):
        score = 0
        state = env.reset()
        finished = False
        
        for j in range(3000):
            state = np.reshape(state, (1, 8))
            if np.random.random() <= epsilon:
                action =  np.random.choice(4)
            else:
                action_values = model.predict(state)
                action = np.argmax(action_values[0])
            next_state, reward, finished, metadata = env.step(action)
            next_state = np.reshape(next_state, (1, 8))
            memory.append((state, action, next_state, reward, finished))
            experience_replay()
            score += reward
            state = next_state
            if finished:
                scores.append(score)
                print(f'Episode = {i}, Score = {score}, Avg_Score = {np.mean(scores[-100:])}')
                break
        
        # Save model check points
        if (i) != 0 and (i) % checkp == 0:
            model.save(f'./saved_models/model_no_tgt_eps_{i}_total_{num_episodes}')
                
    np.savetxt(f'./saved_scores/score_no_tgt_teps{num_episodes}.csv', scores, delimiter=",")

def main():
    if len(sys.argv) <= 2:
        print("Please provide <eps> <checkpoint>")
    else:
        play(num_episodes=int(sys.argv[1]), checkp=int(sys.argv[2]))

if (__name__ == "__main__"):
    main()