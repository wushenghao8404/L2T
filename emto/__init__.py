from gymnasium.envs.registration import register

# register(
#     id='l2t_toy-v0',
#     entry_point='emto.envs.toy_mto_env:ToyTwoTaskOptEnv',
#     # vector_entry_point='',
#     reward_threshold=10000,
#     max_episode_steps=10000,
# )

register(
    id='l2t_emto-v1',
    entry_point='emto.envs.mto_env_v1:MultiTaskOptEnv',
    # vector_entry_point='',
)

# register(
#     id='l2t_base',
#     entry_point='emto.base_env:MultiTaskOptEnv',
#     # vector_entry_point='',
# )

#
# register(
#     id='l2t_emto-v2',
#     entry_point='emto.envs.mto_env_v2:MultiTaskOptEnv',
#     # vector_entry_point='',
# )
#
# register(
#     id='l2t_emto-v3',
#     entry_point='emto.envs.mto_env_v3:MultiTaskOptEnv',
#     # vector_entry_point='',
# )

# register(
#     id='l2t_emto-v4',
#     entry_point='emto.envs.mto_env_v4:MultiTaskOptEnv',
#     # vector_entry_point='',
# )

# register(
#     id='l2t_emto-v5',
#     entry_point='emto.envs.mto_env_v5:MultiTaskOptEnv',
#     # vector_entry_point='',
# )
#
# register(
#     id='l2t_emto-v6',
#     entry_point='emto.envs.mto_env_v6:MultiTaskOptEnv',
#     # vector_entry_point='',
# )

# register(
#     id='l2t_emto-v8',
#     entry_point='emto.envs.mto_env_v8:MultiTaskOptEnv',
#     # vector_entry_point='',
# )
# print('finished importing')