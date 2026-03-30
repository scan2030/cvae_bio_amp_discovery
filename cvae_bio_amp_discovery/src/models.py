"""Model definitions extracted from the notebook."""

from __future__ import annotations

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 3100, num_hidden: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.num_hidden = num_hidden

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_hidden),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.num_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class VAE(AutoEncoder):
    def __init__(self, input_dim: int = 3100, num_hidden: int = 8):
        super().__init__(input_dim=input_dim, num_hidden=num_hidden)
        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = nn.Linear(self.num_hidden, self.num_hidden)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return encoded, decoded, mu, log_var

    def sample(self, num_samples, device=None):
        if device is None:
            device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(num_samples, self.num_hidden).to(device)
            samples = self.decoder(z)
        return samples


class ConditionalVAE(VAE):
    def __init__(
        self,
        num_classes_1,
        num_classes_2,
        num_classes_3,
        num_classes_4,
        num_classes_5,
        input_dim: int = 3100,
        num_hidden: int = 8,
    ):
        super().__init__(input_dim=input_dim, num_hidden=num_hidden)

        self.weight_1 = nn.Parameter(torch.ones(1))
        self.weight_2 = nn.Parameter(torch.ones(1))
        self.weight_3 = nn.Parameter(torch.ones(1))
        self.weight_4 = nn.Parameter(torch.ones(1))
        self.weight_5 = nn.Parameter(torch.ones(1))

        self.label_projector_1 = nn.Sequential(nn.Linear(num_classes_1, self.num_hidden), nn.ReLU())
        self.label_projector_2 = nn.Sequential(nn.Linear(num_classes_2, self.num_hidden), nn.ReLU())
        self.label_projector_3 = nn.Sequential(nn.Linear(num_classes_3, self.num_hidden), nn.ReLU())
        self.label_projector_4 = nn.Sequential(nn.Linear(num_classes_4, self.num_hidden), nn.ReLU())
        self.label_projector_5 = nn.Sequential(nn.Linear(num_classes_5, self.num_hidden), nn.ReLU())

    def condition_on_label_and_features(self, z, y_1, y_2, y_3, y_4, y_5):
        projected_label_1 = self.label_projector_1(y_1.float())
        projected_label_2 = self.label_projector_2(y_2.float())
        projected_label_3 = self.label_projector_3(y_3.float())
        projected_label_4 = self.label_projector_4(y_4.float())
        projected_label_5 = self.label_projector_5(y_5.float())

        return (
            z
            + self.weight_1 * projected_label_1
            + self.weight_2 * projected_label_2
            + self.weight_3 * projected_label_3
            + self.weight_4 * projected_label_4
            + self.weight_5 * projected_label_5
        )

    def forward(self, x, y_1, y_2, y_3, y_4, y_5):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        hid_fea = self.condition_on_label_and_features(z, y_1, y_2, y_3, y_4, y_5)
        decoded = self.decoder(hid_fea)
        return encoded, decoded, mu, log_var, hid_fea

    def sample(self, num_samples, y_1, y_2, y_3, y_4, y_5, device=None):
        if device is None:
            device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(num_samples, self.num_hidden).to(device)
            samples = self.decoder(
                self.condition_on_label_and_features(z, y_1, y_2, y_3, y_4, y_5)
            )
        return samples

    def encode_without_labels(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        return encoded, mu, log_var, z
