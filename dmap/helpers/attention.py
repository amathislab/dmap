import torch
import numpy as np


def compute_attention_score(action_model, obs):
    """Compute the matrix K for a given observation

    Args:
        action_model (DMAPPolicyModel): DMAP model to use
        obs (dict): observation dictionary produced by an RMA environment

    Returns:
        torch.tensor: matrix K of DMAP
    """
    for key, value in obs.items():
        obs[key] = torch.reshape(torch.tensor(value, dtype=torch.float32), (1, -1))
    obs = {"obs": obs}
    with torch.no_grad():
        adapt_input, _ = action_model.get_adapt_and_state_input(obs)
        keys, _ = action_model.get_keys_and_values(adapt_input)
    return keys


def compute_action_without_encoding(
    policy_nets, obs, action_size, embedding_size, device
):
    """For the ablation study: replace the attention encoding with 0s

    Args:
        policy_nets (list[torch.nn.Module]): policy networks of DMAP
        obs (np.array): observation
        action_size (int): size of the action vector
        embedding_size (int): size of the embedding vector
        device (str): "cpu" or "cuda"

    Returns:
        np.array: action computed without the encoding
    """
    state = obs["x_t"]
    action = np.zeros(action_size)
    embedding = np.zeros(embedding_size)
    input_obs = torch.tensor(
        np.concatenate((state, action, embedding)).astype(np.float32)
    ).to(device)
    a_list = []
    with torch.no_grad():
        for n in policy_nets:
            a_list.append(n(input_obs)[0].item())
        action = np.array(a_list)
    return np.clip(action, -1, 1)
