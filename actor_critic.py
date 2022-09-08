from ac_gridworld import *

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

### Model
def return_policy_ac(state_values_ac):
  policy_ac_ = np.zeros_like(game.board.render_np()[0], dtype = str) # TO Store Policy
  for row_idx, row_value_ in enumerate(state_values_ac):
    for col_idx, each_value_ in enumerate(row_value_):
      if each_value_ == -10.0 or each_value_ == 10.0:
        policy_ac_[row_idx][col_idx]= ''
      else:
        up_val = state_values_ac[(row_idx-1),(col_idx)]
        if row_idx < len(state_values_ac) - 1:
          down_val = state_values_ac[(row_idx+1),(col_idx)]

        left_val = state_values_ac[(row_idx),(col_idx-1)]
        if col_idx < len(row_value_) - 1:
          right_val = state_values_ac[(row_idx),(col_idx+1)]

        if row_idx == 0:
          up_val = -10000.0
          if col_idx == 0:
            each_value_ = -10000.0
            left_val = -10000.0
          if col_idx == len(row_value_) - 1:
            right_val = -10000.0
        if row_idx == len(state_values_ac) - 1:
          down_val = -10000.0
          if col_idx == 0:
            left_val = -10000.0
          if col_idx == len(row_value_) - 1:
            right_val = -10000.0
        if col_idx == 0:
          left_val = -10000.0
        if col_idx == len(row_value_) - 1:
          right_val = -10000.0

        d = {'u': up_val, 'd': down_val, 'l': left_val, 'r': right_val}
        direc = max(d, key=d.get)
        policy_ac_[(row_idx, col_idx)] = direc
  return policy_ac_

class Actor(nn.Module):
    def __init__(self, no_states, no_actions, seed):
        super(Actor, self).__init__()
        #self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(no_states, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, no_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.log_softmax(self.fc5(x),dim=-1)
        return x

class Critic(nn.Module):
    def __init__(self, no_states, seed):
        super(Critic, self).__init__()
        #self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(no_states, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

epochs = 5000
losses = []
gamma = 0.9
epsilon = 0.7
seed = 42

actor_net = Actor(36, 4, seed=seed)
critic_net = Critic(36, seed=seed)

game = Gridworld(Rsize=3,Csize=4, mode='static')
state_values_ac = np.zeros_like(game.board.render_np()[0], dtype = float) # TO Store Values of States

state_values_ac[2,3] = 10 # GOAL
state_values_ac[2,2] = -10 # PIT
policy_ac = np.zeros_like(game.board.render_np()[0], dtype = str) # TO Store Policy

def pick_action(state):
        probs = actor_net(state)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, dist

actor_optim = optim.Adam(actor_net.parameters(), lr = 1e-3)
critic_optim = optim.Adam(critic_net.parameters(), lr = 1e-3)
# Replay buffers
actor_replay = []
critic_replay = []
buffer = 80
batchSize =40
earlystop=0.99
earlystop_acc = 0.0
earlystop_decay = 0.9
min_epsilon = 0.1

for i in range(epochs): 

    policy_losses = [] 
    value_losses = []
    episode_reward_list = []
    episode_qval_list = []
    actions_taken = []
    game = Gridworld(Rsize=3,Csize=4, mode='static') # Creating new game for each episode

    state_ = game.board.render_np().reshape(1,36) + np.random.rand(1,36)/10.0 # Initialize the state of the world
    state1 = torch.from_numpy(state_).float() #E State to Tensor

    status = 1 # will be 0 ON EPISODE ENDS
    MAX_MOVES=40
    move_counter = 0
    print_once = 0
    while(status == 1): # Running while loop for each complete episode
       
       player_position = game.board.render_np()[0]
       player_coordinates = tuple(zip(*np.where(player_position == 1)))[0]
       x_coord, y_coord = player_coordinates[0], player_coordinates[1]
       
       orig_reward = game.reward()
       orig_val = critic_net(state1) ## critic1

       action, dist= pick_action(state1)

       rand_val = random.random()

       if (rand_val< epsilon): #I  exploration
            action_ = torch.tensor(np.random.randint(0,4))
            action_choosen = torch.tensor(np.random.randint(0,4))
       else:
            action_= action
            action_choosen = np.max(action.detach().numpy()) #  exploitation

       action = action_set[int(action_choosen)]

       if i%10 == 0:
          if print_once == 0:
            print(f"\n\nEPISODE No.: {i}\n")
            print("ActorCritic on 3*4 GridWorld\n")
            print_once+=1
          actions_taken.append(action)


       game.makeMove(action) #K  now taking action
   
       if i%10 == 0:
          #print(f"\nThe action choosen is:\t{action}\n")
          #print(game.board.render_np()[0]) # printing only Player position
          pass


       state2_ = game.board.render_np().reshape(1,36) + np.random.rand(1,36)/10.0 #  Initializing next state s` after taking that action
       state2 = torch.from_numpy(state2_).float() #L 


       new_val = critic_net(state2)

       new_reward = game.reward() #  reward r(t+1)
    
       if new_reward == -1: # Non-terminal state.
          target = orig_reward + ( gamma * new_val)
       else:
          target = orig_reward + ( gamma * new_reward )
                # In terminal states, the environment tells us
                # the value directly.
                
       best_val = max((orig_val*gamma), target)

       # Now append this to our critic replay buffer.
       critic_replay.append([state1,best_val])
       # If we are in a terminal state, append a replay for it also.
       if new_reward != -1:
         critic_replay.append( [state2, float(new_reward)] )

       actor_delta = new_val - orig_val                
       actor_replay.append([state1, action_, actor_delta])

       # Critic Replays...
       while(len(critic_replay) > buffer): # Trim replay buffer
          critic_replay.pop(0)

       # Start training when we have enough samples.
       if(len(critic_replay) >= buffer):
          minibatch = random.sample(critic_replay, batchSize)
          X_train = []
          y_train = []
          for memory in minibatch:
              m_state, m_value = memory
              y = np.empty([1])
              y[0] = m_value
              X_train.append(m_state.reshape((36,)))
              y_train.append(y.reshape((1,)))
              #X_train = np.array(X_train[0])
             # y_train = np.array(y_train[0])
          y_pred = critic_net(X_train[0])
          y_train = torch.from_numpy(y_train[0])
          loss = F.mse_loss(y_pred.float(), y_train.float())
          critic_optim.zero_grad()
          loss.backward()
          critic_optim.step()

       # Actor Replays...
       while(len(actor_replay) > buffer):
          actor_replay.pop(0)                
       if(len(actor_replay) >= buffer):
          X_train = []
          y_train = []
          minibatch = random.sample(actor_replay, batchSize)
          for memory in minibatch:
             m_orig_state, m_action, m_value = memory
             old_qval = actor_net( m_orig_state.reshape(1,36,) )
             y = np.zeros(( 1, 4 ))
             y[:] = old_qval[:].detach().numpy() 
             # non-standard - decay actions we aren't selecting on this turn.
             #y[:] = old_qval[:] * gamma

             y[0][m_action] = m_value[0]
             X_train.append(m_orig_state.reshape((36,)))
             y_train.append(y.reshape((4,)))
             #X_train = np.array(X_train[0])
            # y_train = np.array(y_train[0])
          y_pred = actor_net(X_train[0])
          y_train = torch.from_numpy(y_train[0])
          loss = F.mse_loss(y_pred.float(), y_train.float())
          actor_optim.zero_grad()
          loss.backward()
          actor_optim.step()

      # print(critic_net(state1))
       state_values_ac[x_coord, y_coord] = max(float(critic_net(state1).data.numpy()), state_values_ac[x_coord, y_coord])
       policy_ac[x_coord, y_coord] = action
       
       #print("loss : \t\t",adv)
       state1 = state2

       if new_reward != -1:
          status = 0
          if i%10 == 0: 
            
            print("STATE VALUES\n")
            print(state_values_ac)
            print("\nPOLICY")
            policy_ac = return_policy_ac(state_values_ac)
            print(policy_ac)
          # Count wins/losses for early-stopping.
          if new_reward == 10: # Win
              earlystop_result = 1.0
          else: # Loss
              earlystop_result = 0.0
          new_acc = ( earlystop_acc * earlystop_decay ) + ((1.0 - earlystop_decay ) * earlystop_result)
          earlystop_acc = new_acc

       #print("loss : \t\t",adv)
       state1 = state2

      
       # Finised Epoch
      # clear_output(wait=True)
      # print("Game #: %s" % (i,))
      # print("Accumulated win percent: %.2f%%" % (earlystop_acc*100) )

       if epsilon > min_epsilon:
          epsilon -= (1/epochs)
       # Check if we can early-stop training.
       if earlystop_acc > earlystop:
          print("Early-Stopping Training")
          break
          
    if epsilon > 0.1: #R  Exploring to Exploiting
        epsilon -= (1/epochs)


print("ActorCritic on 3*4 GridWorld\n")
print("STATE VALUES Actor Critic")
print(state_values_ac)
print("\nPOLICY Actor Critic")
print(policy_ac)