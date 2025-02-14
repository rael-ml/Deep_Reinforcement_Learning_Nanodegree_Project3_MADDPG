# Project Report

## Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Overview

Reinforcement Learning (RL) methods can be categorized into **value-based** and **policy-based** approaches:  
- **Value-based methods** estimate the expected sum of rewards obtained by taking an action in a given state. In deep reinforcement learning, an agent selects the action with the highest **Q-value** (expected cumulative reward).  
- **Policy-based methods** directly predict the best action to take given a state, using a neural network.  

**Deep Deterministic Policy Gradient (DDPG)** is a powerful RL algorithm that combines both approaches. It consists of:  
- **Actor Network (Policy-Based):** Predicts the best action given the current state.  
- **Critic Network (Value-Based):** Evaluates the predicted action to determine its quality.  

By integrating these networks, DDPG enables effective learning in **continuous action spaces**.  

**MADDPG** utilizes multiple agents to apply **Deep Deterministic Policy Gradient**. Each agent's **critic** has access to the states of all agents, allowing it to calculate the action value that leads to the best global cumulative reward. The **actor**, however, only has access to its own state but is trained to predict an action that maximizes the cumulative reward, as determined by the critic.

During training, the agent learns from the critic’s value, which takes into account the states of both agents. However, during testing, the agent uses only its own state, as it has been trained to select actions that maximize the cumulative reward.

**MADDPG** can be used in a collaborative setting, where both agents share the same objective, or in a competitive setting, where the agents compete against each other for control of the environment. The **Tennis** environment is an example of a collaborative scenario, where the agents work together to achieve a common goal.

### Off-Policy Learning  

MADDPG is an **off-policy** algorithm, meaning the learned policy differs from the one used during training. This allows for more efficient experience reuse.  

### Learning Process  

At each step, the agent:  
1. **Stores** the `(state, states_both_agents, action, reward, next_state, nest_state_both_agents, sone)` tuple in the **experience buffer**.  
2. **Samples** a batch from the buffer (once it contains enough experiences).  
3. **Learns** by updating its networks:  
   - The **Actor Network** takes the current state as input and outputs an action.  
   - The **Critic Network** takes the state of both agents and action of the current one as input and estimates the expected reward.  

To stabilize training, both networks have **target networks** (copies of the original networks), which are **soft-updated** at each step to prevent instability.  

### Training the Networks  

#### **Critic Network Update**  
The Critic's error is measured as the difference between:  
- The **Q-value** outputted by the local Critic network (with informantion of both agents).  
- The **Bellman equation** applied to the target network's Q-value:  

$$
Q_{\text{target}} = r + (\gamma \cdot Q_{\text{target, next}} \cdot (1 - \text{done}))
$$

- **$Q_{\text{target}}$** → The **target Q-value**, representing the estimated total reward for a given state-action pair.  
- **$r$** → The **immediate reward** received after taking an action.  
- **$\gamma$** (gamma) → The **discount factor** ($0 < \gamma \leq 1$), which determines how much future rewards influence the current value.  
- **$Q_{\text{target, next}}$** → The **Q-value of the next state**, estimated by the **target Critic network**.  
- **$\text{done}$** → A **binary indicator** (0 or 1) for whether the episode has ended:  
  - **done = 1** → The episode is over (future rewards are ignored).  
  - **done = 0** → The episode is ongoing (future rewards are considered).  
- **$(1 - \text{done})$** → Ensures that if the episode has ended, the **next state's Q-value is ignored** (since there are no future actions).  

This ensures that the local Critic network learns to estimate future expected returns correctly.  

#### **Actor Network Update**  
The Actor’s error is measured by:  
1. Passing the predicted action (from the local Actor network) to the local Critic.  
2. Evaluating whether the Critic assigns a high **Q-value** to the action.  

By optimizing the Actor network to maximize the Critic’s evaluation, the agent learns to take better actions over time.  




### Improvements to the Neural Networks

Despite its efficiency, the simple neural network model is prone to instability. To address this, several improvements have been introduced in the MADDPG:

#### 1. Target Neural Network
In the simple network, updating both the neural network weights and the output values simultaneously leads to instability. A **target network** is used to solve this problem:

- The target network has the same architecture as the main network but with weights "frozen" for a few episodes.
- The main network updates its weights based on the outputs provided by the target network.
- Periodically, the weights of the target network are synchronized with those of the main network, improving stability.
- The Actor has a local and a target network and the Critic also has a local and a taget network.

#### 2. Experience Replay Buffer
To prevent the network from overfitting to sequential patterns in the data, transitions `(s, a, r, s')` are stored in an **experience buffer**:

- Transitions are stored in the buffer.
- During training, random samples from the buffer are used, promoting diversity in the training data and improving efficiency.

#### 3. Ornstein–Uhlenbeck process
Ornstein-Uhlenbeck (OU) noise is added to the network's output to serve as an exploration term. This type of noise is well-suited for continuous action spaces because it is time-correlated, meaning that consecutive actions remain similar rather than changing abruptly. As a result, the model is less likely to propose actions that differ drastically from previous ones, leading to smoother exploration.
In this project, a decay rate was implemented to gradually reduce exploration over time.

## Hyperparameters and Networks Architecture
The table below summarizes the hyperparamters of the MADDPG agent that solved the environment.

| Parameter               | Experiment 1    |
|-------------------------|-----------|
| BUFFER_SIZE            | 100,000   | 
| BATCH_SIZE             | 128        | 
| GAMMA                  | 0.99      | 
| TAU                    | 1.00E-02  | 
| Initial Learning Rate (Actior and Critic)     | 1.00E-02  | 
| Final Learning Rate (Actior and Critic)           |2.00E-04  |  
| Learning Rate Decay           |0.995  | 
| Noise Decay           |0.99  | 
| Weight Decay           |1.00E-05  | 
| n_hidden_layers        | 2         | 
| fc1_units              | 128      | 
| fc2_units              | 128       | 


### Actor and Critic Architectures  

The **Actor network** was structured as follows:  
24 (state size) → 128 → 128 → 2 (action size)

The **Critic network** followed a similar design:  
24 → 128 → 128 → 1

This was the **only combination of hyperparameters** that successfully solved the environment.  

### Hyperparameter Experiments and Observations  

#### 1️⃣ Batch Size  
- Adjusting `batch_size` had **little impact** on performance.  

#### 2️⃣ Networks Architectures  
- Networks with **fewer neurons** (e.g., `64 × 64`) in the hidden layers **failed to solve** the environment.
- Networks with **more neurons** (e.g., `264 × 264`) in the hidden layers also **failed to solve** the environment.  


#### 3️⃣ Decreasing Learning Rate  
Finding the correct learning rate was challenging. After testing many variations of values, implementing a decaying learning rate method seemed to be the most effective. Through trial and error, starting with a learning rate of 1e-2 and decaying it to 2e-4 proved to work well.

#### 4️⃣ Noise Reduction Strategy  
- Implemented a **decaying noise process** to encourage **exploration early** in training while shifting towards **exploitation later**.  
- This approach **improved performance** but was **not sufficient** to solve the environment.  

#### 5️⃣ The Key Breakthrough  
✅ The **key change** that enabled successful training was the **implementation of decaying learning and noise rates**.


Through extensive experimentation, we found that **a well-balanced network architecture and a carefully tuned decaying learning rate** were essential for solving the environment. This was a very simple analysis of the hyperparameters, providing only a preliminary understanding of their behavior. A more detailed investigation, involving extensive exploration of various hyperparameters, repeated experiments, collection of averages, and hypothesis testing, would be necessary to obtain clearer insights into the impact of hyperparameter tuning.

### Reward Plots During Training Across Four Experiments

The environment was successfully solved after 495 episodes, as shown in the figure below:

<img src="rewards.png" alt="Learning Progress" width="400">

The following reward plot illustrates the performance improvements across episodes. This plot highlights how the agent's performance evolved during training.

<img src="rewardsplot.png" alt="Training Reward Plot" width="800">


## Ideas for Future Work
For future improvements, beyond a more in-depth exploration of hyperparameters, we suggest attempting to solve the environment using only the DDPG approach. Since the agents share common objectives, treating both states and actions as a unified entity might also lead to a solution—this approach seems worth exploring.
 
