# Project 2: Continuous Control


This README demonstrates how to train and visualize a solution to the Continuous Project in the Udacity Deep Reinforcement Learning Nanodegree. It is adapted from the course README.

### Getting Started

1. Follow the instructions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up the Deep Reinforcement Learning Nanodegree (DRLND) GitHub repository.

2. Replace the contents of `p2_continuous-control/` folder with this repository.

3. (Linux Only) To operate pytorch and tensorflow with CUDA and cuDNN:

```
conda activate drlnd

pip uninstall tensorflow
conda install -c anaconda tensorflow-gpu

pip uninstall torch
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

4. Download the twenty agent environment from one of the links below. Select the environment that matches your operating system. If using Linux, also download the headless version from [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip):

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
    

5. In the DRLND GitHub repository, create folders `deep-reinforcement-learning/unity_environments/Tennis_Linux` and `deep-reinforcement-learning/unity_environments/Tennis_Linux_NoVis`, place the downloaded environments inside, and unzip the files. 

### Training and Visualization

1. Navigate to `p3_collab-compet/` and run the following:

```
conda activate drlnd
jupyter notebook Tennis.ipynb
```

2. To train the environment without visualization, set the variable **visible_environment** to **False**. <br /> At the top of the jupyter notebook, click *Kernel -> Restart & Run All*. 

3. To train the environment *with* visualization, set the variable **visible_environment** to **True**. 
<br /> At the top of the jupyter notebook, click *Kernel -> Restart & Run All*. 

4. To visualize the trained environment, set the variable **visible_environment** to **True**. 
<br /> At the top of the jupyter notebook, click *Kernel -> Restart*.
<br /> Run each section _**except**_ for **3. Train the Agent with DDPG**

