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
    with torch.no_grad():   # EMA is a weight update — never build a graph here
        return {k: tau * target_params[k] + (1 - tau) * encoder_params[k]
                for k in encoder_params}

# Step 12 - encode_batch
def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    # TODO: Batch-encode observations into latent embeddings using encoder params.
    return encoder_forward(obs, encoder_params)

# Step 13 - init_predictor_params
def init_predictor_params(latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64, seed: int = 0) -> dict:
    # TODO: Initialize and return a dict of MLP predictor parameters.
    torch.manual_seed(seed)
    dico={}
    dico['action_embed_w']=0.02*torch.randn((action_dim,latent_dim)).requires_grad_(True)
    dico['fc1_w']=0.02*torch.randn((hidden_dim,2*latent_dim)).requires_grad_(True)
    dico['fc2_w']=0.02*torch.randn((latent_dim,hidden_dim)).requires_grad_(True)
    dico['fc1_b']=0.02*torch.zeros((hidden_dim)).requires_grad_(True)
    dico['fc2_b']=0.02*torch.zeros((latent_dim)).requires_grad_(True)
    return dico

# Step 14 - embed_action
def embed_action(actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # actions: (B,) long tensor of action indices
    return predictor_params['action_embed_w'][actions]

# Step 15 - predictor_forward
import torch.nn.functional as F
def predictor_forward(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Implement predictor_forward, the forward pass of the action-conditioned dynamics predictor.
    embed_act = embed_action(actions, predictor_params)
 
    concat = torch.cat((embeddings, embed_act), dim=1)
  
    x = F.linear(concat, predictor_params['fc1_w'], predictor_params['fc1_b'])
    x = F.relu(x)
    x = F.linear(x, predictor_params['fc2_w'], predictor_params['fc2_b'])
    
    return x

# Step 16 - predict_next_embedding
def predict_next_embedding(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Predict the next latent embedding given current embeddings and actions...
    forward=predictor_forward(embeddings, actions, predictor_params)
    return forward

# Step 17 - prediction_loss
def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # TODO: Compute the JEPA prediction loss as mean squared error...
    B,D=predicted.shape
    loss=0
    for i in range(B):
        for j in range(D):
            loss+=(predicted[i][j]-target[i][j])**2
    return loss/(B*D)

# Step 18 - variance_loss
def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    # TODO: Compute VICReg variance hinge loss encouraging each dim std >= gamma.
    vector_var=embeddings.var(dim=0)   #vector of size D vector[i] is the variance of the i th dimension
    B,D=embeddings.shape
    vicreg_loss=torch.mean(torch.relu(gamma-torch.sqrt(vector_var+eps)))
    return vicreg_loss

# Step 19 - covariance_loss
import torch

def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    B, D = embeddings.shape
    if D <= 1:
        return torch.tensor(0.0, dtype=embeddings.dtype, device=embeddings.device)

    orig_dtype = embeddings.dtype 
    embeddings_double = embeddings.double()
    centered = embeddings_double - torch.mean(embeddings_double, dim=0)
    cov_matrix = (1 / (B - 1)) * torch.matmul(centered.T, centered)
    mask = ~torch.eye(D, dtype=torch.bool, device=embeddings.device)
    off_diagonal_squares = cov_matrix[mask] ** 2
    cov_loss = off_diagonal_squares.sum() / D
    return cov_loss.to(orig_dtype)

# Step 20 - vicreg_regularizer
def vicreg_regularizer(embeddings: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, gamma: float = 1.0) -> torch.Tensor:
    # TODO: Combine variance and covariance terms into the VICReg regularizer...
    var_l=variance_loss(embeddings, gamma)
    cov_l=covariance_loss(embeddings)
    return var_l*var_weight+cov_l*cov_weight

# Step 21 - jepa_loss
def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, online_embeddings: torch.Tensor, pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    # TODO: Compose the full JEPA objective from prediction and VICReg terms...
    return pred_weight*prediction_loss(predicted,target)+vicreg_regularizer(online_embeddings, var_weight, cov_weight)

# Step 22 - collapse_metric
def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    # TODO: Measure collapse as the mean of per-dimension batch stds
    return torch.mean(torch.std(embeddings,dim=0))

# Step 23 - jepa_training_step
def jepa_training_step(batch, encoder_params, target_params, predictor_params,
                       lr: float = 1e-3, tau: float = 0.99):
  
    enc = {k: v.detach().requires_grad_(True) for k, v in encoder_params.items()}
    pred = {k: v.detach().requires_grad_(True) for k, v in predictor_params.items()}
    tgt = {k: v.detach() for k, v in target_params.items()}

    online = encode_batch(batch['observations'], enc)
    with torch.no_grad():
        target = encode_batch(batch['next_observations'], tgt)

    next_embed_pred = predict_next_embedding(online, batch['actions'], pred)
    loss = jepa_loss(next_embed_pred, target, online)
    loss.backward()

    with torch.no_grad():
        new_enc = {k: (v - lr * v.grad).detach().requires_grad_(True)
                   for k, v in enc.items()}
        new_pred = {k: (v - lr * v.grad).detach().requires_grad_(True)
                    for k, v in pred.items()}

    new_tgt = ema_update(tgt, new_enc, tau)

    loss_value = loss.item()                                  # Python float
    collapse_value = collapse_metric(online.detach()).item()  # Python float, not tensor

    return new_enc, new_tgt, new_pred, loss_value, collapse_value

# Step 24 - train_jepa
def train_jepa(dataset, encoder_params, target_params, predictor_params,
               num_steps: int = 50, batch_size: int = 32, lr: float = 1e-3,
               tau: float = 0.99, seed: int = 0):
    metrics = []
    N = len(dataset['observations'])

    gen = torch.Generator().manual_seed(seed) 

    for j in range(num_steps):
        #random shuffle
        perm = torch.randperm(N, generator=gen)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            batch = {k: v[idx] for k, v in dataset.items()}
            encoder_params, target_params, predictor_params, loss_value, collapse_value = \
                jepa_training_step(batch, encoder_params, target_params, predictor_params, lr, tau)

        metrics.append({'loss': loss_value, 'collapse': collapse_value})

    return encoder_params, target_params, predictor_params, metrics

# Step 25 - rollout_latent_dynamics
def rollout_latent_dynamics(initial_embedding: torch.Tensor,
                            actions: torch.Tensor,
                            predictor_params: dict) -> torch.Tensor:
    if initial_embedding.dim() == 1:
        emb = initial_embedding.unsqueeze(0)
        unbatched = True
    else:
        emb = initial_embedding
        unbatched = emb.shape[0] == 1       

    if actions.dim() == 0:
        actions = actions.view(1, 1)
    elif actions.dim() == 1:
        actions = actions.unsqueeze(0)     

    B, D = emb.shape
    T = actions.shape[-1]
    if actions.shape[0] == 1 and B > 1:      
        actions = actions.expand(B, -1)

    traj = [emb]
    for t in range(T):
        emb = predict_next_embedding(emb, actions[:, t], predictor_params)
        traj.append(emb)

    trajectory = torch.stack(traj, dim=0)    

    return trajectory[:, 0, :] if unbatched else trajectory

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

