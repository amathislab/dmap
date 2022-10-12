import torch
import numpy as np
from torch.utils.data import TensorDataset


class AdaptModule(torch.nn.Module):
    """Wrapper for the TCN of the RMA policy"""

    def __init__(self, fc_layers, conv_layers, logits):
        super().__init__()
        self.fc_layers = fc_layers
        self.conv_layers = conv_layers
        self.logits = logits

    def forward(self, x):
        out = self.fc_layers(x).transpose(1, 2)
        out = self.conv_layers(out)
        out = self.logits(out)
        return out


def build_dataset(saver, model_phase_1, device="cuda"):
    encoder_input_list = []
    adapt_input_list = []
    for transition in saver.data:
        obs = transition.obs
        e_t = obs["e_t"].astype(np.float32)
        encoder_input_list.append(e_t)
        obs_size = np.prod(obs["x_t"].shape)
        action_size = np.prod(obs["a_t"].shape)
        x_prev = obs["x_prev"].astype(np.float32).reshape((-1, obs_size))
        a_prev = obs["a_prev"].astype(np.float32).reshape((-1, action_size))
        adapt_input = np.concatenate((x_prev, a_prev), axis=1)
        adapt_input_list.append(adapt_input)

    encoder_input_tensor = torch.tensor(encoder_input_list, device=device)
    adapt_input_tensor = torch.tensor(adapt_input_list, device=device)
    with torch.no_grad():
        encoder_features = model_phase_1._encoder_hidden_layers(encoder_input_tensor)
        encoder_output_tensor = model_phase_1._encoder_logits(encoder_features)
    print("Dataset tensors shape:")
    print(encoder_input_tensor.shape)
    print(adapt_input_tensor.shape)
    print(encoder_output_tensor.shape)
    return TensorDataset(adapt_input_tensor, encoder_output_tensor)


def transfer_policy_weights(trainer_phase_1, trainer_phase_2):
    weights_2 = trainer_phase_2.get_weights()["default_policy"]
    weights_1 = trainer_phase_1.get_weights()["default_policy"]
    for key in weights_2.keys():
        if "policy" in key:
            assert key in weights_1, weights_1.keys()
            weights_2[key] = weights_1[key]

    trainer_phase_2.set_weights({"default_policy": weights_2})


def train_loop(dataloader, model, loss_fn, optimizer, verbose=True):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0 and verbose:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss
