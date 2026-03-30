"""Training and visualization helpers extracted from the notebook."""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import VAE, ConditionalVAE
from preprocessing import AMINO_ACIDS, index_to_aa_mapping


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def loss_function(recon_x, x, mu, logvar, input_dim=3100):
    bce = F.binary_cross_entropy(
        recon_x.view(-1, input_dim),
        x.view(-1, input_dim),
        reduction="sum",
    )
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def visualize_embeddings(model, X_train, y_train, device, batch_size=1000):
    model.eval()

    if not isinstance(X_train, torch.Tensor):
        X_train_tensor = torch.from_numpy(X_train).float().to(device)
    else:
        X_train_tensor = X_train.float().to(device)

    with torch.no_grad():
        mu_list = []
        for i in range(0, len(X_train_tensor), batch_size):
            batch = X_train_tensor[i : i + batch_size]
            mu = model.encoder(batch)
            mu_list.append(mu.cpu().numpy())

    embeddings = np.concatenate(mu_list, axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=1, alpha=0.7, c=y_train)
    plt.colorbar(scatter)
    plt.title("2D Visualization of AutoEncoder Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()


def train_vae(X_train, learning_rate=1e-3, num_epochs=100, batch_size=32, device=None, input_dim=3100):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(X_train, torch.Tensor):
        X_train_tensor = torch.from_numpy(X_train).float().to(device)
    else:
        X_train_tensor = X_train.float().to(device)

    model = VAE(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="sum")
    model.to(device)

    train_loader = torch.utils.data.DataLoader(X_train_tensor, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            _, decoded, mu, log_var = model(data)
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = criterion(decoded, data) + 3 * kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}: loss={epoch_loss:.4f}")

    return model


def train_cvae(
    X_train,
    y_1_train,
    y_2_train,
    y_3_train,
    y_4_train,
    y_5_train,
    num_classes_1,
    num_classes_2,
    num_classes_3,
    num_classes_4,
    num_classes_5,
    learning_rate=1e-3,
    num_epochs=10,
    batch_size=32,
    device=None,
    input_dim=3100,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(X_train, torch.Tensor):
        X_train_tensor = torch.from_numpy(X_train).float().to(device)
    else:
        X_train_tensor = X_train.float().to(device)

    y_1_train_tensor = torch.nn.functional.one_hot(
        torch.from_numpy(y_1_train).long(), num_classes=num_classes_1
    ).float().to(device)
    y_2_train_tensor = torch.nn.functional.one_hot(
        torch.from_numpy(y_2_train).long(), num_classes=num_classes_2
    ).float().to(device)
    y_3_train_tensor = torch.nn.functional.one_hot(
        torch.from_numpy(y_3_train).long(), num_classes=num_classes_3
    ).float().to(device)
    y_4_train_tensor = torch.nn.functional.one_hot(
        torch.from_numpy(y_4_train).long(), num_classes=num_classes_4
    ).float().to(device)
    y_5_train_tensor = torch.nn.functional.one_hot(
        torch.from_numpy(y_5_train).long(), num_classes=num_classes_5
    ).float().to(device)

    model = ConditionalVAE(
        num_classes_1,
        num_classes_2,
        num_classes_3,
        num_classes_4,
        num_classes_5,
        input_dim=input_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    train_loader = torch.utils.data.DataLoader(
        list(
            zip(
                X_train_tensor,
                y_1_train_tensor,
                y_2_train_tensor,
                y_3_train_tensor,
                y_4_train_tensor,
                y_5_train_tensor,
            )
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in range(num_epochs):
        total_loss = 0.0
        for data, y_1, y_2, y_3, y_4, y_5 in train_loader:
            data = data.to(device)
            y_1 = y_1.to(device)
            y_2 = y_2.to(device)
            y_3 = y_3.to(device)
            y_4 = y_4.to(device)
            y_5 = y_5.to(device)

            _, decoded, mu, log_var, _ = model(data, y_1, y_2, y_3, y_4, y_5)
            loss = loss_function(decoded, data, mu, log_var, input_dim=input_dim)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}: loss={epoch_loss:.4f}")

    return model


def extract_conditioned_z(
    model,
    X,
    y_1,
    y_2,
    y_3,
    y_4,
    y_5,
    num_classes,
    batch_size=32,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes_1, num_classes_2, num_classes_3, num_classes_4, num_classes_5 = num_classes
    model.eval()

    if not isinstance(X, torch.Tensor):
        X_tensor = torch.from_numpy(X).float().to(device)
    else:
        X_tensor = X.float().to(device)

    y_1_tensor = torch.nn.functional.one_hot(torch.from_numpy(y_1).long(), num_classes=num_classes_1).float().to(device)
    y_2_tensor = torch.nn.functional.one_hot(torch.from_numpy(y_2).long(), num_classes=num_classes_2).float().to(device)
    y_3_tensor = torch.nn.functional.one_hot(torch.from_numpy(y_3).long(), num_classes=num_classes_3).float().to(device)
    y_4_tensor = torch.nn.functional.one_hot(torch.from_numpy(y_4).long(), num_classes=num_classes_4).float().to(device)
    y_5_tensor = torch.nn.functional.one_hot(torch.from_numpy(y_5).long(), num_classes=num_classes_5).float().to(device)

    data_loader = torch.utils.data.DataLoader(
        list(zip(X_tensor, y_1_tensor, y_2_tensor, y_3_tensor, y_4_tensor, y_5_tensor)),
        batch_size=batch_size,
        shuffle=False,
    )

    conditioned_z_list = []
    with torch.no_grad():
        for data, y_1_batch, y_2_batch, y_3_batch, y_4_batch, y_5_batch in data_loader:
            data = data.to(device)
            y_1_batch = y_1_batch.to(device)
            y_2_batch = y_2_batch.to(device)
            y_3_batch = y_3_batch.to(device)
            y_4_batch = y_4_batch.to(device)
            y_5_batch = y_5_batch.to(device)
            _, _, _, _, conditioned_z = model(
                data, y_1_batch, y_2_batch, y_3_batch, y_4_batch, y_5_batch
            )
            conditioned_z_list.append(conditioned_z.cpu().numpy())

    return np.vstack(conditioned_z_list)


def extract_z(model, X, batch_size=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    if not isinstance(X, torch.Tensor):
        X_tensor = torch.from_numpy(X).float().to(device)
    else:
        X_tensor = X.float().to(device)

    z_list = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            _, _, _, z = model.encode_without_labels(batch)
            z_list.append(z.cpu().numpy())

    return np.vstack(z_list)


def reconstruct_sequence(decoded_output, max_length, min_probability=0.05):
    num_amino_acids = len(AMINO_ACIDS)
    reconstructed_sequences = []
    index_to_aa = index_to_aa_mapping(AMINO_ACIDS)

    if isinstance(decoded_output, torch.Tensor):
        decoded_output = decoded_output.cpu().detach().numpy()

    for one_hot_vector in decoded_output:
        one_hot_matrix = one_hot_vector.reshape(-1, num_amino_acids)
        amino_acid_sequence = []
        for one_hot in one_hot_matrix:
            aa_index = int(np.argmax(one_hot))
            if one_hot[aa_index] > min_probability:
                amino_acid_sequence.append(index_to_aa[aa_index])
            else:
                amino_acid_sequence.append("-")

        reconstructed_sequences.append("".join(amino_acid_sequence).rstrip("-"))

    return reconstructed_sequences
