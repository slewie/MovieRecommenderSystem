import torch.nn as nn


class DenseAutoencoder(nn.Module):
    """
    Class that creates embedding for movies, it accepts the converted movie title and genres vector
    """
    def __init__(self, input_size, emb_size=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, emb_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        embeddings = self.encoder(x)
        reconstructions = self.decoder(embeddings)
        return reconstructions

    def encode(self, x):
        return self.encoder(x)
