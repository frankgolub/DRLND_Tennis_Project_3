This report presents a solution to the Navigation Project in the Udacity Deep Reinforcement Learning Nanodegree. 

## Problem Statement

The following is sligthly modified from the course materials:

> In this environment, twenty identical double-jointed arms can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
>
> The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
>
> To solve this environment, the average score of all twenty arms must be greater than 30 over 100 episodes.

<p align="center">
  <img width="460" height="300" src="Results/Reacher_soln.png">
</p>
<!-- https://gist.github.com/DavidWells/7d2e0e1bc78f4ac59a123ddf8b74932d -->

<p align="center">
  <em> <b> Fig. 1. </b> Twenty double-jointed arms reach the target green locations.</em>
</p>


This repository presents a solution in 289 episodes using a Deep Deterministic Policy Gradient approach. The code and network architectures are adapted from a Udacity-supplied solution to the OpenAI-gym Pendulum environment. Fig. 1 depicts twenty double-jointed arms that learn to reach the target green locations. Videos before and after training can be seen [here](https://youtu.be/Lok1AF4mFmE) and [here](https://youtu.be/xpmJBSXJaD8) respectively. Fig. 2 shows the score increasing stochastically as the model trains.

The network trains in 55 min. 53 s. using an Intel i9-9900K 8-core CPU, 64 GB of RAM, and an 11GB NVIDIA 2080 TI GPU.

<p align="center">
  <img width="460" height="300" src="Results/Figure_ddpg_normal_soln_final.png">
</p>

<p align="center">
  <em> <b> Fig. 2. </b> The average score between episodes 190 and 289 is 30.06.</em>
</p>

## Deep Deterministic Policy Gradient 

Given a state **s**, a DDPG trains two networks, an actor and a critic, off-policy to deterministically estimate the continuous action **a** and the action-value function **Q**, respectively, that  maximize the expected discounted reward. The reward obtained after taking action **a** is used to calculate the gradient of the critic, and this gradient is used to calculate the gradient of the actor. For training, experiences are buffered, randomized, and batched together. Google DeepMind introduced the concept of a [DDPG](https://arxiv.org/abs/1509.02971) in 2016.

### Network Architectures and Hyperparameters

The input to the architecture of the actor is a 33-element state vector. This architecture consists of a 33 x 400 fully-connected layer, a batch-normalization function, a relu activation function, a 400 x 300 fully-connected layer, a relu activation function, a 300 x 4 fully-connected layer, and a tanh activation function. The weights of the last fully-connected layer are initialized randomly in the range [-3e-3, 3e-3].

The input to the architecture of the critic is the same 33-element state vector. This architecture also consists of a 33 x 400 fully-connected layer, a batch-normalization function, and a relu activation function. However, the output of this relu function is concatenated with the 4-element action-value estimated from the first architecture to make a 404 element vector. Applied to this vector is a 404 x 300 fully-connected layer, a relu-activation function, and a 300 x 1 fully-connected layer. The weights of the last fully-connected layer are also initialized randomly in the range [-3e-3, 3e-3].

Key hyperparameters that represent the network include buffer size (1e6), batch size (64), discount factor (0.99), and tau (1e-3). The networks use Adam for gradient descent. The learning rates of both the critic and the actor were set to 1e-3 for the first 200 episodes and 1e-4 for the next 40. Both the learning rate and the momentum for the two networks were arrested for the last 49 so that the agent could obtain the requisite average score over the preceeding 100 episodes.

One episode corresponds to 1400 time steps. At each time step, twenty agents simultaneously add their experiences to the same replay buffer, and the regular weights of both networks update only once. The previous weights of the target network add to the updated weights of the regular network by the ratio of tau:(1 - tau) to form the new weights of the target network.

## Lessons Learned

The choice of random seed substantially affects whether a network can learn. Neither network invokes as many batch normalization functions as the DeepMind paper describes. For example, batch normalization is _not_ used after the state input. The hyperparameters in this implementation are similar to but not the same as the hyperparameters in the DeepMind implementation. Both the learning rate and the momentum can be stopped to assess performance over a period of time. 

## Future Work

[Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html) is a modified version of DDPG that is supposedly less dependent on hyperparameter tuning. To verify this assertion, I would like to apply the the OpenAI implementation to this problem. Even small changes to hyperparameters in my submitted DDPG solution drastically affect performance.

## References

Lillicrap,  T. P.,  Hunt,  J. J.,  Pritzel,  A.,  Heess,  N.,  Erez,T., Tassa, Y., Silver, D., and Wierstra, D.  [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971). _arXiv preprint arXiv:1509.02971, 2016._

Fujimoto, S., van Hoof, H., Meger, D. [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf). _arXiv preprint arXiv:1802.09477, 2018._



