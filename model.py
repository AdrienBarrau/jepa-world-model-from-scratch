"""
JEPA World Model from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_env_state
def init_env_state(room_size: int = 8, seed: int | None = None) -> torch.Tensor:
    # TODO: Initialize a random agent position state inside a square 2D room...
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    return torch.randint(low=0, high=room_size, size=(2,), generator=gen).float()

# Step 2 - apply_action
def apply_action(state: torch.Tensor, action: int, room_size: int = 8) -> torch.Tensor:
    # TODO: Apply a discrete action to the agent state with wall clamping.
    if action==0:
        if state[1]>0:
            state[1]-=1
    if action==1:
        if state[1]<room_size-1:
            state[1]+=1
    
    if action==3:
        if state[0]<room_size-1:
            state[0]+=1
    if action==2:
        if state[0]>0:
            state[0]-=1
    return state

# Step 3 - render_observation
def render_observation(state: torch.Tensor, room_size: int = 8) -> torch.Tensor:
    # TODO: Render a (1, room_size, room_size) float32 obs with agent pixel 1.0...
    res=torch.zeros((room_size,room_size),dtype=torch.float32)
    ind_i=int(state[1])
    ind_j=int(state[0])
    res[ind_i][ind_j]=1.0
    return res.unsqueeze(0)

# Step 4 - env_reset
def env_reset(room_size: int = 8, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Reset the environment by sampling a new state and its observation.
    if seed is not None:
        state=init_env_state(room_size, seed)
    else:
        state=state=init_env_state(room_size)
    return state,render_observation(state, room_size)

# Step 5 - env_step
def env_step(state: torch.Tensor, action: int, room_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Advance the env by one action; return (next_state, next_observation).
    next_state=apply_action(state,action,room_size)
    next_observation=render_observation(state, room_size)
    return (next_state, next_observation)

# Step 6 - collect_random_transitions
import random
def collect_random_transitions(num_transitions, room_size=8, seed=0):
    N = num_transitions
    rng = random.Random(seed)

    dico = {
        'observations':       torch.zeros((N, 1, room_size, room_size)),
        'actions':            torch.zeros((N,), dtype=torch.long),
        'next_observations':  torch.zeros((N, 1, room_size, room_size)),
        'states':             torch.zeros((N, 2)),
        'next_states':        torch.zeros((N, 2)),
    }

    state, obs = env_reset(room_size, seed)
    for i in range(N):
        action = rng.randint(0, 3)
        state_before = state.clone()
        next_state, next_obs = env_step(state, action, room_size)

        dico['observations'][i]        = obs
        dico['actions'][i]             = action
        dico['next_observations'][i]   = next_obs
        dico['states'][i]              = state_before
        dico['next_states'][i]         = next_state

        state, obs = next_state, next_obs

    return dico

# Step 7 - build_transition_dataset
def build_transition_dataset(num_transitions: int = 512, room_size: int = 8, seed: int = 0) -> dict:
    # TODO: Build a JEPA training-ready transition dataset by collecting random transitions...
    return collect_random_transitions(num_transitions, room_size, seed)

# Step 8 - init_encoder_params
import torch
def init_encoder_params(obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32, seed: int = 0) -> dict:
    # TODO: Initialize parameters of a small CNN encoder that maps pixel observations to latent embeddings.
    #parameter initialization
    torch.manual_seed(seed)
    conv1_w=torch.randn((16,obs_channels,3,3)).requires_grad_(True)*0.1
    conv2_w=torch.randn((32,16,3,3)).requires_grad_(True)*0.1
    conv1_b=torch.zeros((16)).requires_grad_(True)
    conv2_b=torch.zeros((32)).requires_grad_(True)
    fc_w=torch.randn((latent_dim,32 * (room_size // 2) * (room_size // 2))).requires_grad_(True)*0.1
    fc_b=torch.zeros((latent_dim)).requires_grad_(True)
    dico={'conv1_w':conv1_w, 
    'conv1_b':conv1_b, 
    'conv2_w':conv2_w, 
    'conv2_b':conv2_b, 
    'fc_w':fc_w, 
    'fc_b':fc_b}
    return dico

# Step 9 - encoder_forward
def encoder_forward(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    # TODO: Run a small two-layer CNN that maps pixel observations to latent embeddings.
    conv1=F.conv2d(obs,encoder_params['conv1_w'],encoder_params['conv1_b'],
    stride=1,padding=1)
    x=F.relu(conv1)
    conv2=F.conv2d(x,encoder_params['conv2_w'],encoder_params['conv2_b'],
    stride=2,padding=1)
    x=F.relu(conv2)
    flat=torch.flatten(x,start_dim=1)
    fc=F.linear(flat,encoder_params['fc_w'],encoder_params['fc_b'])

    return fc

# Step 10 - init_target_encoder
def init_target_encoder(encoder_params: dict) -> dict:
    # TODO: Create the EMA target encoder by deep-copying the online encoder params.
    dico={key:value.clone().detach() for key,value in encoder_params.items()}
    return dico

# Step 11 - ema_update
def ema_update(target_params: dict, encoder_params: dict, tau: float = 0.99) -> dict:
    # TODO: Refresh target encoder params via EMA of the online encoder.
    target_encoder=init_target_encoder(encoder_params)
    for key in target_encoder.keys():
        target_encoder[key]=tau*target_params[key]+(1-tau)*target_encoder[key]
    return target_encoder

# Step 12 - encode_batch
def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    # TODO: Batch-encode observations into latent embeddings using encoder params.
    return encoder_forward(obs, encoder_params)

# Step 13 - init_predictor_params (not yet solved)
# TODO: implement

# Step 14 - embed_action (not yet solved)
# TODO: implement

# Step 15 - predictor_forward (not yet solved)
# TODO: implement

# Step 16 - predict_next_embedding (not yet solved)
# TODO: implement

# Step 17 - prediction_loss (not yet solved)
# TODO: implement

# Step 18 - variance_loss (not yet solved)
# TODO: implement

# Step 19 - covariance_loss (not yet solved)
# TODO: implement

# Step 20 - vicreg_regularizer (not yet solved)
# TODO: implement

# Step 21 - jepa_loss (not yet solved)
# TODO: implement

# Step 22 - collapse_metric (not yet solved)
# TODO: implement

# Step 23 - jepa_training_step (not yet solved)
# TODO: implement

# Step 24 - train_jepa (not yet solved)
# TODO: implement

# Step 25 - rollout_latent_dynamics (not yet solved)
# TODO: implement

# Step 26 - multi_step_prediction_error (not yet solved)
# TODO: implement

# Step 27 - init_linear_probe (not yet solved)
# TODO: implement

# Step 28 - train_linear_probe (not yet solved)
# TODO: implement

# Step 29 - probe_state_recovery (not yet solved)
# TODO: implement

# Step 30 - encode_goal (not yet solved)
# TODO: implement

# Step 31 - latent_cost (not yet solved)
# TODO: implement

# Step 32 - sample_action_sequences (not yet solved)
# TODO: implement

# Step 33 - score_action_sequences (not yet solved)
# TODO: implement

# Step 34 - select_best_plan (not yet solved)
# TODO: implement

# Step 35 - mpc_step (not yet solved)
# TODO: implement

# Step 36 - run_mpc_episode (not yet solved)
# TODO: implement

# Step 37 - evaluate_planner (not yet solved)
# TODO: implement

# Step 38 - jepa_world_model_experiment (not yet solved)
# TODO: implement

