from gym.envs.registration import register

register(
    id='one-random-evader-v0',
    entry_point='gym_pursuit_evasion.envs:OneRandomEvaderEnv',
)

register(
    id='one-stationary-evader-v0',
    entry_point='gym_pursuit_evasion.envs:OneStationaryEvaderEnv',
)