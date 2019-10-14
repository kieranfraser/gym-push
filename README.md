# Gym-push
### A custom OpenAI Gym environment for intelligent push-notifications

#### Content:
1. [Quick start](#quick-start)
2. [Running basic environment](#Running-basic-environment)
3. [EvalUMAP Challenge 2019](#Evalumap-challenge-2019) 
4. [Task description](#Task-description)
5. [Data explained](#Data-explained)
6. [Running task 1](#Running-task-1)
7. [Running task 2](#Running-task-2)
8. [Submitting results](#Submitting-results)
9. [Contact details](#Contact-details)

> Gym-push environments shall be used for comparative evalutation in the [EvalUMAP 2020](http://evalumap.adaptcentre.ie/) challenge.

## Quick start
Install via pip:
```sh
> pip install gym-push
```

## Running basic environment
The *basic-v0* environment simulates notifications arriving to a user in different contexts. The features of the context and notification are simplified. 

Below is an example of setting up the basic environment and stepping through each moment (context) a notification was delivered and taking an action (open/dismiss) upon it. For demonstration purposes, this *agent* randomly samples an action from the action_space. Intelligent agents will use the features of the context and notification to identify optimal performance over time.

Agent performance is evaluated by comparing the action chosen with the *groundtruth* action and monitoring the subsequent *Click-Through-Rate* of notifications.
```sh
import gym
env = gym.make('gym_push:basic-v0')
obs = env.reset()
finished = False
total_reward = 0
while not finished:
	obs, reward, finished, info = env.step(env.action_space.sample())
	total_reward = total_reward + reward
	env.render()
print('Total Reward: ', total_reward)
```

## EvalUMAP Challenge 2019
To be added.

## Task Description
#### Task 1

![Task 1 operation diagram](docs/img/task_1_diagram.jpg)

#### Task 2

![Task 2 operation diagram](docs/img/task_2_diagram.jpg)

## Data explained
To be added.

## Running task 1
To be added.

## Running task 2
To be added.

## Submitting results
To be added.

## Contact details
The following is a list of people who are available to answer queries:

|Name|Email|Regarding|
|-------------|-----------------------------|-------------------|
|Kieran Fraser|kieran.fraser@adaptcentre.ie | Gym-push          |
|Bilal Yousuf |bilal.yousuf@adaptcentre.ie  | EvalUMAP Challenge|
