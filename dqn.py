from dqn_gridworld import *
import torch
import random
import numpy as np
from matplotlib import pylab as plt

### Model
l1 = 36
l2 = 64
l3 = 48
l4 = 16
l5 = 4

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4),
    torch.nn.ReLU(),
    torch.nn.Linear(l4,l5),

)

learning_rate = 1e-3
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#displaying start state
game = Gridworld(Rsize=3,Csize=4, mode='static') #C  Creating new game for each episode
state_ = game.board.render_np().reshape(1,36) #D  Initialize the state of the world\
print("Start State of the GridWorld\n")
print(game.board.render_np()) # Printing PLAYER P, GOAL +, PIT -, WALL W.
print("\n\nTraining Begins\n\n")

epochs = 1001
losses = []
counter = 0 
gamma = 0.9
epsilon = 0.4


state_values_dqn = np.zeros_like(game.board.render_np()[0], dtype = float)
state_values_dqn[2,2] = -10 # PIT
state_values_dqn[2,3] = +10 # GOAL
policy_dqn = np.zeros_like(game.board.render_np()[0], dtype = str)

for i in range(epochs): #B 
    episode_reward_list = []
    episode_qval_list = []
    game = Gridworld(Rsize=3,Csize=4, mode='static') #C  Creating new game for each episode
    state_ = game.board.render_np().reshape(1,36) + np.random.rand(1,36)/10.0 #D  Initialize the state of the world
    state1 = torch.from_numpy(state_).float() #E State to Tensor
    status = 1 #F
    print_once = 0
    while(status == 1): #G  Running while loop for 1 complete episode
        player_position = game.board.render_np()[0]
        player_coordinates = tuple(zip(*np.where(player_position == 1)))[0]
        x_coord, y_coord = player_coordinates[0], player_coordinates[1]

        qval = model(state1) #H  predicting value of state using DQN model
        qval_ = qval.data.numpy()
        episode_qval_list.append(np.max(qval_))
        if (random.random() < epsilon): #I  exploration
            action_ = np.random.randint(0,4) 
        else:
            action_ = np.argmax(qval_) #  exploitation
            action = action_set[action_] #J refer to action_set dict
            policy_dqn[x_coord, y_coord] = action

        state_values_dqn[x_coord, y_coord] = max(qval_[0][action_]/2, state_values_dqn[x_coord, y_coord])

        action = action_set[action_] #J refer to action_set dict
        #policy[x_coord, y_coord] = action
        
        if i%1000 == 0:
          if print_once == 0:
            print(f"\n\n EPISODE No.: {i}\n")
            print_once+=1
          print(f"\nThe action choosen is:\t{action}\n")


        game.makeMove(action) #K  now taking action

        
        if i%1000 == 0:
          print('\n')
          print(player_position)
          print('\n') # printing only Player position
        state2_ = game.board.render_np().reshape(1,36) + np.random.rand(1,36)/10.0 #  Initializing next state s`
        state2 = torch.from_numpy(state2_).float() #L 
        reward = game.reward() #  reward r(t+1)
        episode_reward_list.append(reward)

        with torch.no_grad():
            newQ = model(state2.reshape(1,36))  # we see the values output by model are the q_vals
        maxQ = torch.max(newQ) #M  maxQ at s`
        if reward == -1: #N  Updating the value of the state using current reward and value of s`
            Y = reward + (gamma * maxQ)
        else:
            Y = reward
        Y = torch.Tensor([Y]).detach() # q_val updated one : r(t+1) + γmaxQA(S(t+1))
        X = qval.squeeze()[action_] #O predicted one using the model

        loss = loss_fn(X, Y) #P

        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        state1 = state2  #  State update
        if reward != -1: #Q  Episode Ending
          status = 0
          if i%1000 == 0:
            #print(f"\tReward Sequence:\t{episode_reward_list}")
            print(f"\nSTATE_VALUES\n", state_values_dqn)
            print(f"POLICY\n", policy_dqn)
            print(f"\nSum Total Rewards:\t{sum(episode_reward_list)}")          
          
    if epsilon > 0.1: #R  Exploring to Exploiting
        epsilon -= (1/epochs)

print("DQN on 3*4 GridWorld\n")
print("STATE VALUES DQN")
print(state_values_dqn)
print("\nPOLICY DQN")
print(policy_dqn)
