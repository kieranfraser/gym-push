from gym.envs.registration import register

register(
    id='basic-v0',
    entry_point='gym_push.envs:Basic',
)
register(
    id='evalumap1-v0',
    entry_point='gym_push.envs:EvalUMAP1',
)