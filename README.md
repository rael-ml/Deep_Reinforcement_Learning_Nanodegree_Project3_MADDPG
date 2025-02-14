# MADDPG Application with Unity ML-Agents for Udacity's Deep Reinforcement Learning Nanodegree

## Project Overview
In this project, the goal is to train two agents using ** Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** to play a game of tennis as long as possible.

![Reacher Environment](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)

## Problem Description
The agent interacts with a virtual environment where the **state space** consists of **24 dimensions**, including position  and velocity of the ball and racket. 

Based on these observations, the agent must decide how to move to reach the ball and throw it to the other agent.

The **action space** consists of **two continuous actions**, representing the movement toward the net and jumping, with values ranging from **-1 to 1**.  

- The agent receives a **reward of +0.1** for each time step it reach the ball.
- The objective is to keep the playing without letting the ball falling.   
- The task is **episodic**, meaning each episode has a defined start and end.  
- The environment is considered **solved** when the agent achieves an **average score of +0.5 over 100 consecutive episodes**.  

## Files Included
- **MADDPG Projct.ipynb**: Jupyter Notebook containing the implementation of the DDPG agent for the Reacher environment.
- **README.md**: file with details about the project and setup instructions.
- **checkpoint_actor_1.pth**: Saved weights of the first agent's trained Actor network that can solve the environment.
- **checkpoint_critic_1.pth**: Saved weights of the first agent's trained Critic network that can solve the environment.
- **checkpoint_actor_2.pth**: Saved weights of the second agent's trained Actor network that can solve the environment.
- **checkpoint_critic_2.pth**: Saved weights of the second agent's trained Critic network that can solve the environment.
- **requirements.txt**: Python dependencies required to run the project.
- **setup.py**: Script for installing the Unity Machine Learning Agents library and related dependencies.
- **Report.md**: A detailed report on the learning algorithm, hyperparameters, and ideas for future improvements.
- **rewardsplot.png**: Plot of the rewards of the experiment detailed in Report.md.
- **rewards.png**: Print of the rewards of the experiment detailed in Report.md.

## Getting Started

To run the code, you'll need to download the tennis environment from one of the following links based on your operating system:

- **Linux**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- **Mac OSX**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- **Windows (32-bit)**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- **Windows (64-bit)**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Once downloaded, unzip the file and place it in the project folder.

## Installation Instructions

1. First, install the necessary dependencies by running the following command:

<pre> python  !pip -q install . </pre>

2. This will trigger the setup.py script and install the required Python libraries listed in requirements.txt.

3. After installation, restart the kernel in your Jupyter Notebook environment.

4. Open MADDPG Project.ipynb to begin running the MADDPG agent on the Tennis environment.
