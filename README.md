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
The use-case for the proposed challenge is personalized mobile phone notification generation. Previous work in this space has explored intercepting incoming mobile notifications, mediating their delivery such that irrelevant or unnecessary notifications do not reach the end-user and generating synthetic notification datasets from real world usage data. The next step toward an improved notification experience is to generate personalised notifications in real-time, removing the need for interception and delivery mediation. 

Specifically, assuming individuals’ interactions with their mobile phone have been logged, the challenge is to create an approach to generate personalized notifications on individuals’ mobile phones, whereby such personalization would consist of deciding what events (emails, alerts, reminders etc.) to show to the individual and when to show them. Given the number of steps associated with such personalization, the task proposed in this paper will focus on the first step in this process, that of user model generation using the logged mobile phone interactions. For this task a dataset consisting of several individuals’ mobile phone interactions is provided, described in [Data explained](#Data-explained).

## Task Description
#### Task 1
<p align="center">
  <img width="50%" height="50%" src="docs/img/task_1_diagram.jpg">
</p>

The diagram above illustrates the operation flow for Task 1. 

Participants query *gym-push* for 3 months of historical data. Using this data, the participants should create a user model which takes a context (e.g. Time: 'morning', Place: 'airport', etc.) as input and outputs a personalized notification (e.g. App: 'news', Subject: 'weather', etc.) for the given context. Gym-push can be used for training purposes by returning performance metrics for notifications generated using the training contexts.

Once the model is built, it can be evaluated, again using gym-push. This is achieved by requesting *test* data, 3 months of contextual evaluation data, from gym-push. Resulting notifications generated should then be returned to the environment where evaluation metrics are calculated using test assets.


#### Task 2
<p align="center">
  <img width="50%" height="50%" src="docs/img/task_2_diagram.jpg">
</p>

The diagram above illustrates the differing operation flow for Task 2. Participants are asked to create a user model based on the same notification, context and engagement features - but without historical notification data to train with (although, they can query the environment for sample data with which to create their model). 

In contrast to Task 1, this user model will need to query the gym-push environment at each step to receive a current context feature and a previous user notification-engagement feature. As the environment steps through each context item and as engagement history becomes available, the user model can exploit this information to improve the generation of personalized notifications.

The goal is to develop a model which adapts and learns how to generate personalized notifications in real-time, without prior history of the user (cold-start problem). Evaluation is continuous for this task and a summary of results is issued once all context features have been processed.

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
