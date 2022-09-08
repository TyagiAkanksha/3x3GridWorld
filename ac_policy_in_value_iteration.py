## USING ACTOR-CRITIC POLICY IN VI

from actor_critic import *

print("AC on 3*4 GridWorld\n")
print("STATE VALUES AC")
print(state_values_ac)
print("\nPOLICY AC")

print(policy_ac)

policy_coords = dict()
for row_idx, row_value_ in enumerate(policy_ac):
  for col_idx, each_value_ in enumerate(row_value_):
    if each_value_ != '':
      policy_coords[(row_idx, col_idx)] = each_value_
print(policy_coords)

import numpy as np
from scipy.sparse import dok_matrix

'''==================================================
Initial set up
=================================================='''

#Hyperparameters
SMALL_ENOUGH = 0.01
GAMMA = 0.9         
NOISE = 0.10  

#Define all states
all_states=[]
for i in range(3): #rows
    for j in range(4): #cols
            all_states.append((i,j))

#Define rewards for all states
rewards = {}
for i in all_states:
    if i == (2,2):
        rewards[i] = -10.0000
    elif i == (2,3):
        rewards[i] = 10.0000
    else:
        rewards[i] = -1

#Dictionnary of possible actions. We have one "end" state (2,2)
actions = policy_coords

#Define an initial policy
policy={}
for s in actions.keys():
    policy[s] = actions[s]

#Define initial value function 
V={}
for s in all_states:
    if s in actions.keys():
        V[s] = 0
    if s ==(2,2):
        V[s] = -10
    if s == (2,3):
        V[s] = 10

'''==================================================
Value Iteration
=================================================='''

iteration = 0

while True:
    biggest_change = 0
    for s in all_states:            
        if s in policy:
            
            old_v = V[s]
            new_v = 0
            
            for a in actions[s]:
                if a == 'u':
                    nxt = [s[0]-1, s[1]]
                if a == 'd':
                    nxt = [s[0]+1, s[1]]
                if a == 'l':
                    nxt = [s[0], s[1]-1]
                if a == 'r':
                    nxt = [s[0], s[1]+1]

                #Choose a new random action to do (transition probability)
                random_1= ([i for i in actions[s] if i != a]) ############################################################
                if random_1 == 'u':
                    act = [s[0]-1, s[1]]
                if random_1 == 'd':
                    act = [s[0]+1, s[1]]
                if random_1 == 'l':
                    act = [s[0], s[1]-1]
                if random_1 == 'r':
                    act = [s[0], s[1]+1]

                #Calculate the value
                nxt = tuple(nxt)
                #act = tuple(act)
                v = rewards[s] + (GAMMA * ((1-NOISE)* V[nxt])) #+ (NOISE * V[act])
                if v > new_v: #Is this the best action so far? If so, keep it
                    new_v = v
                    policy[s] = a

       #Save the best of all actions for the state                                
            V[s] = round(new_v, 4)
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))
            
   #See if the loop should stop now         
    if biggest_change < SMALL_ENOUGH:
        #print(policy)
        #print(V)
        break
    iteration += 1


# Convert Coordinate Dictionary to Matrix
def matrix_display(test_dict, which="State_Values"):
    # Using loop + max() + list comprehension
    temp_x = max([cord[0] for cord in test_dict.keys()])
    temp_y = max([cord[1] for cord in test_dict.keys()])
    res = [[0] * (temp_y + 1) for ele in range(temp_x + 1)]
    
    for (i, j), val in test_dict.items():
        res[i][j] = val
    
    # printing result 
    print(f"\n{which}")
    for row in res:
      print(row)
    print('\n') 

print("Value Iteration on 3*4 GridWorld USING AC POLICY")
matrix_display(V, which="State_Values")

