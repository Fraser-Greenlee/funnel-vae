import torch
from torch import nn


class LatentEncoderNTokens(nn.Module):
    '''
        Converts N hidden tokens into N seperate latent codes.
    '''
    def __init__(self, config):
        super().__init__()
        self.token_to_latent = nn.Linear(config.t5.d_model, config.latent_size)
        self.n_tokens = config.n_latent_tokens
        self.tanh = nn.Tanh()

    def forward(self, encoding) -> torch.Tensor:
        return self.tanh(self.token_to_latent(encoding))[:, : self.n_tokens, :]


VAE_ENCODER_MODELS = {
    "": LatentEncoderNTokens,
}
