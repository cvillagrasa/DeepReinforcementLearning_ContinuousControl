# Deep Reinforcement Learning Nanodegree

This repository contains my **Deep Reinforcement Learning** solution to one of 
the stated problems within the scope of the following Nanodegree:

[Deep Reinforcement Learning Nanodegree Program - Become a Deep Reinforcement Learning Expert](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)



Deep Reinforcement Learning is a thriving field in AI with lots of practical 
applications. It consists of the use of Deep Learning techniques to be able to
solve a given task by taking actions in an environment, in order to achieve a 
certain goal. This is not only useful to train an AI agent so as to play videogames, 
but also to set up and solve any environment related to any domain. In particular, 
as a Civil Engineer, my main area of interest is the **AEC field** (Architecture, 
Engineering & Construction).

## Continuous Control Project

This project consists of a Continuous Control problem, which is contained within the Policy 
Based Methods chapter and is solved by means of Deep Deterministic Policy Gradient algorithms (DDPG and TD3).

### Environment description

The environment used for this project is based on the *Unity ML-Agents* Reacher 
environment from 2018. Nonetheless, the updated corresponding equivalent in 2022 
would be the
[Worm Environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#worm).

It consists of a playground with a certain number of double-jointed arms in it (one for the single-agent version
or 20 for the multi-agent version). In addition, there are some sphere-shaped volumes, which move over time, designing 
the goal location for the arms' hand. The goal is then for the arms to maintain its position at the target location 
for as many time steps as possible.

![alt text](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)
<br/>

Its characteristics are as follows:
- The state space has 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.
- The action space consists of 4 continuous actions with component values between -1 and +1. They correspond to the 
amount of torque applicable to both joints.
- A reward of +0.1 is provided for each step that the agent's hand is in the goal location.

### Getting Started

Firstly, the reader is advised to set up the 
[Worm Environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#worm)
instead of the legacy Reacher one, specially if not following the Nanodegree. This will allow to install recent 
versions of the libraries and potentially avoid some unfixed bugs from old versions.

#### Installing required dependencies

Having said that, the exact environment used in this repository -for the scope of the Nanodegree- can be setup by
following the instructions from the "Dependencies" header on the link below, which becomes *Step 1*:
[udacity/deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning#dependencies)

As *Step 2*, the compressed file corresponding to the user operating system must be downloaded from one of the links 
below, and placed inside the `./p2_continuous-control/` folder after unzipping it. At least the multi-agent version is 
needed to reproduce the trainings of this repository. In case both single and multi-agent are downloaded, rename the 
folders as needed. These newly created paths will be set up on the next step.
- Linux [single_agent]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Linux [multi-agent]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX [single-agent]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Mac OSX [multi-agent]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit) [single-agent]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (32-bit) [multi-agent]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit) [single-agent]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
- Windows (64-bit) [multi-agent]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

At this point, *Step 3* requires to clone the current repository and place the folder at the root of the freshly 
configured *deep-reinforcement-learning*. From there, it only remains to change the environment path(s) to match the 
file(s) downloaded in *Step 2*. This path(s) can be changed at 
[the following lines of environment.py](https://github.com/cvillagrasa/DeepReinforcementLearning_ContinuousControl/blob/166943cb7c2f399ec54a35ed47763143c149e606/environment.py#L14-L15).

Lastly, *Step 4* consists of installing Seaborn within the Python environment by running `pip install seaborn`.

And that's it. After those four steps, the Jupyter Notebook *Continuous_Control.ipynb* from this repository can already 
be executed.

### Solution

My solution to the posed problem can be found in the 
[Solution Report](https://htmlpreview.github.io/?https://github.com/cvillagrasa/DeepReinforcementLearning_ContinuousControl/blob/master/Report.html), 
as well as in the code included 
within this repository.

It is worth stating that it has been a fascinating exercise which has let me further understand the implications of 
multi-agent training, as well as the dynamics of the Deep Deterministic Policy Gradient algorithm (DDPG), along with 
its supercharged Twin Delayed variant (TD3).
