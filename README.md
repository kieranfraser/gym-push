# Gym-push: A custom OpenAI Gym environment for intelligent push-notifications
> Gym-push environments shall be used for comparative evalutation in the [EvalUMAP 2020](http://evalumap.adaptcentre.ie/) challenge.

#### Content:
1. [Quick start](#quick-start)
2. [Running basic environment](#running-basic-environment)
3. [EvalUMAP Challenge 2019](#evalumap-challenge-2019) 
4. [Task description](#task-description)
5. [Data explained](#data-explained)
6. [Running task 1](#running-task-1)
7. [Running task 2](#running-task-2)
8. [Submitting results](#submitting-results)
9. [Contact details](#contact-details)


## 1. Quick start
Install via pip:
```sh
> pip install gym-push
```

## 2. Running basic environment
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

## 3. EvalUMAP Challenge 2019
To be added.

## 4. Task Description
To be added.

## 5. Data explained
To be added.

## 6. Running task 1
To be added.

## 7. Running task 2
To be added.

## 8. Submitting results
To be added.

## 9. Contact details
The following is a list of people who are available to answer queries:

|Name|Email|Regarding|
|-------------|-----------------------------|-------------------|
|Kieran Fraser|kieran.fraser@adaptcentre.ie | Gym-push          |
|Bilal Yousuf |bilal.yousuf@adaptcentre.ie  | EvalUMAP Challenge|
