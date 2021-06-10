#!/usr/bin/env python3

from pathlib import Path

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from pennylane.templates import AmplitudeEmbedding

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm import tqdm

from datasets import make_blobs, rotate_2d


dev = qml.device("default.qubit", wires=2)


def layer(W):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.CNOT(wires=[0, 1])


@qml.qnode(dev)
def circuit(weights, features):
    AmplitudeEmbedding(features=features, wires=range(2))

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))


def vqc(var, features):
    weights = var[0]
    bias = var[1]
    return circuit(weights, features) + bias


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss


def cost(weights, features, labels):
    predictions = [vqc(weights, f) for f in features]
    return square_loss(labels, predictions)


def rotate_and_train(degrees: float = 0.0) -> dict:
    rotated_X = X if np.allclose(degrees, 0.0) else rotate_2d(X, origin, degrees)

    padding = 0.3 * np.ones((len(X), 1))
    X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
    norm = np.linalg.norm(X_pad, axis=1)
    X_norm = X_pad / norm[:, None]

    sample_train, sample_test, label_train, label_test = train_test_split(
        X_norm, y, test_size=0.2, shuffle=True, random_state=params["random_state"]
    )

    num_train = len(label_train)
    num_qubits = params["features"]
    num_layers = 3
    var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)

    opt = NesterovMomentumOptimizer(0.01)
    batch_size = 5

    var = var_init
    for it in range(60):
        batch_index = np.random.randint(0, num_train, (batch_size,))
        sample_train_batch = sample_train[batch_index]
        label_train_batch = label_train[batch_index]
        var = opt.step(lambda v: cost(v, sample_train_batch, label_train_batch), var)

    label_predict = np.array([np.sign(vqc(var, s)) for s in sample_test])
    precision, recall, fscore, _ = precision_recall_fscore_support(
        label_test, label_predict
    )
    accuracy = accuracy_score(label_test, label_predict)

    data = {
        "Angle": degrees,
        "Precision_0": precision[0],
        "Recall_0": recall[0],
        "Fscore_0": fscore[0],
        "Precision_1": precision[1],
        "Recall_1": recall[1],
        "Fscore_1": fscore[1],
        "Accuracy": accuracy,
    }
    return data


origin = (0.0, 0.0)

print("Changing distributions\n")
distributions = ["gumbel", "laplace", "logistic", "normal", "vonmises"]

directory = "results"
Path(directory).mkdir(parents=True, exist_ok=True)
output = directory + "/vqc_amp_enc_distributions_summary.csv"

summary_tables = []
for distribution in tqdm(distributions):

    params = {
        "samples": 60,
        "centers": 2,
        "features": 2,
        "random_state": 11,
        "center_box": (-7.5, 7.5),
    }

    X, y = make_blobs(
        n_samples=params["samples"],
        n_features=params["features"],
        distribution=distribution,
        centers=params["centers"],
        center_box=params["center_box"],
        random_state=params["random_state"],
    )

    y = 2 * y - np.ones(len(y))

    rows = []
    for angle in tqdm(np.arange(0.0, 375.0, 15.0)):
        rows.append(rotate_and_train(angle))

    columns = list(rows[0].keys())
    df = pd.DataFrame(rows, columns=columns)
    df.set_index(columns[0], inplace=True)
    summary_tables.append(df)

summary_df = pd.concat(summary_tables, axis=1, keys=distributions)
summary_df.to_csv(output)


print("Changing sigmas\n")
distribution = "normal"
sigmas = np.linspace(0.1, 2, 10)

output = directory + "/vqc_amp_enc_sigmas_summary.csv"

summary_tables = []
for sigma in tqdm(sigmas):

    params = {
        "samples": 60,
        "centers": 2,
        "features": 2,
        "random_state": 11,
        "sigma": sigma,
        "center_box": (-7.5, 7.5),
    }

    X, y = make_blobs(
        n_samples=params["samples"],
        n_features=params["features"],
        distribution=distribution,
        centers=params["centers"],
        cluster_std=params["sigma"],
        center_box=params["center_box"],
        random_state=params["random_state"],
    )

    rows = []
    for angle in tqdm(np.arange(0.0, 375.0, 15.0)):
        rows.append(rotate_and_train(angle))

    columns = list(rows[0].keys())
    df = pd.DataFrame(rows, columns=columns)
    df.set_index(columns[0], inplace=True)
    summary_tables.append(df)

summary_df = pd.concat(summary_tables, axis=1, keys=sigmas)
summary_df.to_csv(output)


print("Changing centers\n")
distribution = "normal"
centers = [-2, 0, 2]

output = directory + "/vqc_amp_enc_centers_summary.csv"

summary_tables = []
for center in tqdm(centers):

    params = {
        "samples": 60,
        "centers": [(center, center), (center, center)],
        "features": 2,
        "random_state": 11,
        "sigma": 1.0,
        "center_box": (-7.5, 7.5),
    }

    X, y = make_blobs(
        n_samples=params["samples"],
        n_features=params["features"],
        distribution=distribution,
        centers=params["centers"],
        cluster_std=params["sigma"],
        center_box=params["center_box"],
        random_state=params["random_state"],
    )

    rows = []
    for angle in tqdm(np.arange(0.0, 375.0, 15.0)):
        rows.append(rotate_and_train(angle))

    columns = list(rows[0].keys())
    df = pd.DataFrame(rows, columns=columns)
    df.set_index(columns[0], inplace=True)
    summary_tables.append(df)

summary_df = pd.concat(summary_tables, axis=1, keys=centers)
summary_df.to_csv(output)
