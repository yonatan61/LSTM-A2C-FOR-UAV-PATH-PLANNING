"""
!sudo apt-get update
!pip install 'imageio==2.4.0'
!pip install pyvirtualdisplay
!pip install pyglet
!pip install imageio-ffmpeg
"""

import gym
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import base64
import imageio
import IPython
import PIL.Image
import pyvirtualdisplay
import numpy as np
import matplotlib.patches as patches
from collections import defaultdict
from moviepy.video.io.bindings import mplfig_to_npimage

class enviornment:
  def __init__(self):
    print('initialize env')
    self.grid= np.zeros((21, 21))
    self.total_user_distribution=self.users_generator()
    self.state=self.reset() #for q1 
    


  def users_generator(self):
    self.total_user_distribution=[]
    for i in [[17,4,5,40],[12,6,4,15],[6,15,4,15],[15,15,5,20],[7,16,3,20]]:
      
      x,y,std,num=i[0],i[1],i[2],i[3]
      X=np.abs(np.random.normal(x,std,num))  #by using abs all the users with -ve will be refelcted 
      Y=np.abs(np.random.normal(y,std,num))
      user_distribution=[]

      #refelect back the users during distribution  
      for k, (i,j) in enumerate(zip(X,Y),):
        if i >20:
          X[k]=20-(i-20)
        if j>20:
          Y[k]=40-j 

      #ignore users thare are in the landing and no fly zone  
      for i,j in zip(X,Y):
        if i>=7 and i<= 9 and j>=18 :
          continue
        if i>=8 and i<= 14 and j>=10 and j<=13 :
          continue
        user_distribution.append((i,j))
      self.total_user_distribution.append(user_distribution)
    
    return self.total_user_distribution
  
  #creat image
  def render(self,Title="",state=(0,0),optimal_path=[],path_plot=False):
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"]=(10,9)
    ax.imshow(self.grid, cmap='Greys',origin='lower')
    rect = patches.Rectangle((7, 18), 2, 2, linewidth=2, edgecolor='black', facecolor='green')
    ax.add_patch(rect) 
    rect = patches.Rectangle((8, 10), 6, 3, linewidth=2, edgecolor='black', facecolor='red')
    ax.add_patch(rect) 
    plt.text(9.5, 11.2, "No Fly Zone", color='Black', fontsize=14)
    plt.text(7.7, 18.7, "SL", color='Black', fontsize=14)
    plt.xticks(np.arange(0, 21, 1),np.arange(0, 21, 1))
    plt.yticks(np.arange(0, 21, 1),np.arange(0, 21, 1))
    plt.title(Title)
    plt.grid()


    for t,symbol in zip(self.total_user_distribution,['x','o','*','#','+']):
        for i in t:
          plt.text(i[0], i[1], symbol, color='Black', fontsize=11)

    rect = patches.Rectangle((state[0]-1.5, state[1]-1.5), 3, 3, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    if path_plot:
      for state, next_state in optimal_path:
        plt.arrow(state[0],state[1],0.9*(next_state[0]-state[0]), 0.9*(next_state[1]-state[1]),head_width=0.1, color='Red',width=0.01)

    #plt.show()
    plt.close()
    return fig
  
  

  def is_outoff_grid(self,i, j):
      if i < 0 or i > 20 or j < 0 or j > 20:
          return True
      elif i>=8 and i<= 14 and j>=10 and j<=13 :
          return True
      else:
          return False

  def surveillance(self,total_user_distribution,xnow,ynow,xnext,ynext,history):
  
    coverd_users_now=[]
    coverd_users_next=[]
    if (xnow,ynow) not in history:
        
        xmax=xnow+1.5
        xmin=xnow-1.5
        ymax=ynow+1.5
        ymin=ynow-1.5
        for t in self.total_user_distribution:
          for i in t:
            
            if i[0]<=xmax and i[0]>=xmin and i[1]<=ymax and i[1]>=ymin:
              coverd_users_now.append((i[0],i[1]))
    
    if (xnext,ynext) not in history:
        
        xmax=xnext+1.5
        xmin=xnext-1.5
        ymax=ynext+1.5
        ymin=ynext-1.5
        for t in self.total_user_distribution:
          for i in t:
            
            if i[0]<=xmax and i[0]>=xmin and i[1]<=ymax and i[1]>=ymin:
              coverd_users_next.append((i[0],i[1]))
              
              #print(i[0],i[1])
    return len(list(set(coverd_users_next) - set(coverd_users_now)))  #when overlap only count the new people found


  def step(self,state, action):
        
        self.battery+=-1
        captured_ppl=0
        # take one step
        i, j = state[0]+3*action[0], state[1]+3*action[1]

        # check if the next state is out off grid
        if self.is_outoff_grid(i, j):
            
            #if it is in no fly zone  it will take its to  previous state
            if i>=8 and i<= 14 and j>=10 and j<=13 :
              i, j = state[0], state[1]
            else :
              i, j=np.abs(i), np.abs(j)   # if it is out of grid it will be refelcted
              if i >20:
                i=40-i
              if j>20:
                j=40-j

          
            
            
            reward=-5
            if self.battery==0:
              self.Done=True 
            self.state=(i,j)
            return (i,j), reward,self.Done,captured_ppl

        if i >=7 and i <= 9 and j>=18 and j<=20 :
            reward = 20
            self.Done=True
            
            self.state=(i,j)
            return (i,j), reward,self.Done,captured_ppl
        else:
            captured_ppl=self.surveillance(self.total_user_distribution,state[0],state[1],i,j,self.history)  #current state and next state will be consdiered
            reward =captured_ppl-1
            if (i,j) in self.history:
              reward=-10
            if self.battery==0:
              self.Done=True 
        self.state=(i,j)
        return (i,j), reward,self.Done,captured_ppl

  def reset(self):
    #self.state=(0,0) #q1
    Xint=np.random.choice([0, 1, 2])  #intial randomly selected [0,2]
    Yint=np.random.choice([0, 1, 2])

    self.state=(Xint,Yint)  #for Q2,Q3,Q4
    self.battery=30
    self.Done=False
    self.history=[]
    
    return self.state

#creat video
def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)

def create_policy_eval_video(policy,agent, filename,env, num_episodes=5, fps=3,):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      state = env.reset()
      
      done = False
      while not done:
        
        action = actions[policy(np.array(state), agent)]                               
        state,reward,done, _ = env.step(state,action)
        
        x=env.render('DDQN',state)
        numpy_fig = mplfig_to_npimage(x)
        video.append_data(numpy_fig)
        
  return embed_mp4(filename)

actions=([(0, 1), (0,-1), (-1,0), (1,0)])  #up,down,left,right
#video=imageio.get_writer("final.mp4", fps=3)

#################################################################
#Hyperparameters
batch_size = 64  # @param {type:"integer"}

#episode options
max_episodes = 1000
num_play_episodes = 10  # @param {type:"integer"}
num_target_episodes = 10  # @param {type:"integer"}
eval_interval=20  # evaluation interval

# learning options
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.999 # was 0.99 # discount_factor
epsilon = 1.0 # starting exploration rate
epsilon_step = 10/max_episodes # epsilon step size
min_epsilon = 0.05 # min exploration rate

#optimizer options:
loss_object = tf.keras.losses.MeanSquaredError()
opt1 = "tf.keras.optimizers.Adam(learning_rate = 0.001)"
#opt2 = "tf.keras.optimizers.SGD(learning_rate = 0.001)"


# experience options
num_fetch_experiences = 64 # number of experiences to fetch
num_setup_experiences = int(num_fetch_experiences*2) # number of initial episodes to store was 100
num_store_experiences = num_fetch_experiences
exp_buffer_len = int(num_setup_experiences*2) # was 100000

############################################################################
# DDQN AGENT

class DDQN_Agent:
    
    def __init__(self):
        self.train_Q = self._build_model() #Q Network
        self.target_Q = self._build_model() #Target Network
        
    # copy weights of train_q network to target_q network
    def update_target_Q(self):
        self.target_Q.set_weights(self.train_Q.get_weights())
        
    # trains agent by calculating loss(TD error) following the present policy    
    def train(self, batch, use_ddqn=True):
        state_b, action_b, reward_b, new_state_b, done_b = batch #sampled from reply buffer
        batch_len = np.array(state_b).shape[0]
        state_b = tf.convert_to_tensor(state_b, dtype=tf.float32)
        
        # use training network to get Qvalues for a state
        trainQ_b = self.train_Q(state_b)
        targetQ_b = np.copy(trainQ_b)
        
        new_state_b = tf.convert_to_tensor(new_state_b, dtype=tf.float32)
        
        next_targetQ_b = self.target_Q(new_state_b)       
        #used for ddqn
        next_trainQ_b = self.train_Q(new_state_b)
        # select max next state value
        max_next_q_b = np.amax(next_targetQ_b, axis=1)
        if (use_ddqn==True):
            max_next_action_b = np.argmax(next_trainQ_b, axis=1)
            for i in range(batch_len):
                max_next_q_b[i] = next_targetQ_b[i][max_next_action_b[i]]       
        # update target_q value with next state value from target network
        for i in range(batch_len):
            target_q_val = reward_b[i]
            if not done_b[i]:
                target_q_val += gamma * max_next_q_b[i]
            targetQ_b[i][action_b[i]] = target_q_val
        # fit the training network to match estimated reward for next state
        train_hist = self.train_Q.fit(x=state_b, y=targetQ_b, verbose=0)
        loss = train_hist.history['loss']
        return loss
    
    # Define helper for hidden Dense layers with desired activation and init
    def dense_layer(self, num_units):
        return Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))
    
    # define helper function for output layer with desired activation and init
    def q_values_layer(self, num_actions):
        return Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
    
    # build the model
    def _build_model(self):
        model = Sequential()
        model.add(self.dense_layer(100))
        model.add(self.dense_layer(50))
        model.add(self.q_values_layer(4))  #num of actions
        
        
        print('ADAM')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), loss='mse')
        
        #model.summary()
        return model

#################################################

# POLICIES

''' My policy code here '''
class Policy:
    def __init__(self):
        # self.epsilon = epsilon
        self.epsilon=0.1
    
    # random policy independent of state
    def random(self, state, agent):
        # use random policy
        action = np.random.choice([0, 1, 2, 3])  #gives index of the action
        return action
    
    def update_epsilon(self, min_epsilon, epsilon_step):
        if (self.epsilon > min_epsilon):
            self.epsilon -= epsilon_step
    
    #epsilon greedy policy determines best action per state
    def e_greedy(self, state, agent):
        # use random policy epsilon percent of times
        x = np.random.random()
        if (x < self.epsilon):
            action =  np.random.choice([0, 1, 2, 3])
        else:
            action = self.greedy(state, agent)
        return action
        
    
    # greedy policy selects best action
    def greedy(self, state, agent):
        # identify all possible actions for each state
        state_tf = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        #state_tf = tf.convert_to_tensor(state, dtype=tf.float32)
        state_Q = agent.train_Q(state_tf)
        action = np.argmax(state_Q) # use in case multiple maximums
        return action

########################################################
# DATA COLLECTION
# REPLAY MEMORY BUFFER
class Experience_Buffer:
    
    # initialize
    def __init__(self):
        self.exp_buffer = deque(maxlen=exp_buffer_len)
        
    # store experiences
    def exp_store(self, environment, agent, policy, num_experiences):
      
      for i in range(num_experiences): 
          state=environment.state
          action_index=policy.e_greedy(np.array(state), agent)  #modification
          action = actions[action_index]
                
          new_state, reward, done, captured_ppl = environment.step(state, action)
                
          self.exp_buffer.append((state, action_index, reward, new_state, done))
          if done:
            environment.reset()
             
      #policy.update_epsilon(min_epsilon, epsilon_step)
       
      

    # fetch experiences
    def exp_fetch(self, num_experiences):
        exp_batch = random.sample(self.exp_buffer, num_experiences)
        state_b, action_b, reward_b, new_state_b, done_b = [],[],[],[],[]
        for exp_sample in exp_batch:
            state_b.append(exp_sample[0])
            action_b.append(exp_sample[1])
            reward_b.append(exp_sample[2])
            new_state_b.append(exp_sample[3])
            done_b.append(exp_sample[4])
        return state_b, action_b, reward_b, new_state_b, done_b

#####################################################################
# METRICS AND EVALUATION

def calc_avg_return(environment, agent, policy, play_episodes):

  total_return = 0.0

  total_captured_ppl=0
  all_optima_path=[]
  temp_captured_ppl=[]    #used to identify the optimal captured among the episdoes
  for i in range(play_episodes):

    state = environment.reset()
    episode_return = 0.0
    done = False
    steps = 0
    path = []
    
    
    tem_comparision=0
    


    while not done:
      action = actions[policy.greedy(np.array(state), agent)]
      new_state, reward, done, captured_ppl = environment.step(state,action)
      
      total_captured_ppl+=captured_ppl
      #print("is it done",done)
      path.append([state, new_state])

      episode_return += reward
      state = new_state
      steps += 1

      tem_comparision+=captured_ppl  #to see the optimal path

    all_optima_path.append(path)
    temp_captured_ppl.append(tem_comparision)


    total_return += episode_return
    #max_steps = max(max_steps, steps)
    print('Game=',i,'Played steps=',steps,'Battery left',environment.battery)
  
  index_maximum_captured=np.argmax(np.array(temp_captured_ppl))
  optimal_path=all_optima_path[index_maximum_captured]

  avg_return = total_return / play_episodes
  avg_ppl=total_captured_ppl/play_episodes

  return avg_return,avg_ppl,optimal_path

def plot_results(x,y,ylabel,title,eval_interval=1):
    # Visualization: Plots
    x_data = range(0, x,eval_interval)
    
    plt.plot(x_data, y)

    plt.ylabel(ylabel)
    plt.xlabel('episodes')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()




###### TRAIN THE AGENT #######
def train_model(num_episodes,eval_interval):
    env = enviornment()
    agent = DDQN_Agent()
    env.reset()
    policy = Policy()
    buffer = Experience_Buffer()
    loss_b = np.zeros(num_episodes)
    avg_rtn_b=[]
    avg_ppl_total=[]
    
    #title of video
    env.render()

    filename='Trained_DDQN_Adam'
    

    # initialize the experience buffer with e-greedy policy
    print('initialize the experience buffer with {:d} experiences\n'.format(num_setup_experiences))
    buffer.exp_store(env, agent, policy, num_setup_experiences)  #buffer holds= state,action_index, reward, new_state, done
    
    # begin running through episodes
    print('begin running for {:d} episodes\n'.format(num_episodes))
    for episode in range(num_episodes):
        # save a new set of experiences using e-greedy policy
        buffer.exp_store(env, agent, policy, num_store_experiences)
        # randomly select a batch of stored experiences
        exp_batch = buffer.exp_fetch(num_fetch_experiences)
        # retrain model with new batch of experiencesS
        loss = agent.train(exp_batch)#############################
        
        # evaluate performance of the new model using greedy policy
        if (episode+1) % eval_interval == 0:
          avg_return, avg_ppl,optimal_path = calc_avg_return(env, agent, policy, num_play_episodes)

          avg_rtn_b.append(np.asarray(avg_return))
          avg_ppl_total.append(np.asarray(avg_ppl))
          print('episode {:d} of {:d}, people captured: {:0.4f}, loss: {:0.4f}'.format(episode,max_episodes,avg_ppl,loss[0]))
        # save loss and return
        loss_b[episode] = np.asarray(loss)
        
        #max_steps_b[episode] = np.asarray(max_steps)
        print('episode {:d} of {:d},loss: {:0.4f}'.format(episode,max_episodes,loss[0]))
        if episode % num_target_episodes == 0:
            print('\tupdating target weights...')
            agent.update_target_Q()
    create_policy_eval_video(policy.greedy,agent, filename,env)
    fig=env.render(optimal_path=optimal_path,path_plot=True)

    tf.keras.backend.clear_session()
    return (avg_rtn_b, loss_b,avg_ppl_total,optimal_path,fig)

##########################################
# main 

''' train the model '''
print('begin training the model!\n')

titleM = 'ddqn model'
Return={}


titleO = ', opt-Adam'
print('running with Adam optimizer')
title = titleM + titleO

avg_returns, losses, avg_ppl_total,optimal_path,fig = train_model(max_episodes,eval_interval)
Return['ddqn']=avg_returns
print('done! Plotting results...')

plot_results(max_episodes,avg_ppl_total,'Captured_People',title,eval_interval)
plot_results(max_episodes,avg_returns,'average_returns',title,eval_interval)
plot_results(max_episodes,losses,'losses',title)
print('all done! Enjoy your chips!\n')
fig