#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from datasets import make_blobs, rotate_2d


def rotate_and_train(degrees: float = 0.0) -> None:
    rotated_X = X if np.allclose(degrees, 0.0) else rotate_2d(X, origin, degrees)

    # plt.scatter(
    #     x=rotated_X[:, 0][y == 0],
    #     y=rotated_X[:, 1][y == 0],
    #     c="orange",
    #     s=250,
    #     edgecolors="k",
    # )
    # plt.scatter(
    #     x=rotated_X[:, 0][y == 1],
    #     y=rotated_X[:, 1][y == 1],
    #     c="red",
    #     s=250,
    #     edgecolors="k",
    # )
    # plt.grid(color="black", linestyle="-", linewidth=2)
    # plt.legend(
    #     labels=["Cluster 1", "Cluster 2"],
    #     loc=0,
    #     ncol=2,
    #     fancybox=True,
    #     shadow=True,
    #     fontsize=28,
    # )
    # image_dir = directory + "/images"
    # p = Path(image_dir)
    # p.mkdir(parents=True, exist_ok=True)
    # plt.savefig(image_dir + f"/data_{degrees}_degrees.png", dpi=90)
    # plt.cla()

    sample_train, sample_test, label_train, label_test = train_test_split(
        rotated_X, y, test_size=0.2, shuffle=True, random_state=params["random_state"]
    )
    class_labels = [r"B", r"A"]
    training_dataset = {
        key: np.array(sample_train[label_train == k, :])[:]
        for k, key in enumerate(class_labels)
    }
    test_dataset = {
        key: np.array(sample_test[label_test == k, :])[:]
        for k, key in enumerate(class_labels)
    }

    feature_map = ZZFeatureMap(feature_dimension=params["features"], reps=5)
    optimizer = SPSA(maxiter=40, c0=4.0, skip_calibration=True)
    var_form = TwoLocal(
        num_qubits=params["features"],
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz",
        reps=3,
    )
    vqc = VQC(
        optimizer=optimizer,
        feature_map=feature_map,
        var_form=var_form,
        training_dataset=training_dataset,
        test_dataset=test_dataset,
    )
    backend = BasicAer.get_backend("qasm_simulator")
    quantum_instance = QuantumInstance(
        backend,
        shots=1024,
        seed_simulator=params["random_state"],
        seed_transpiler=params["random_state"],
    )
    result = vqc.run(quantum_instance)
    with open(output, "a") as f:
        f.write(f"Rotation: {degrees} degrees\n")
        f.write(
            classification_report(
                label_test, vqc.predict(sample_test, quantum_instance)[1]
            )
        )
        f.write("\n")


origin = (0.0, 0.0)

distributions = ["gumbel", "laplace", "logistic", "lognormal", "normal", "vonmises"]

for distribution in distributions:
    directory = f"results/{distribution}"
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)

    output = directory + "/vqc.txt"

    with open(output, "w") as f:
        f.write(f"QSVM with {distribution} distribution\n")
        f.write("=============================\n")
        f.write("\n")

    params = {
        "samples": 60,
        "centers": 2,
        "features": 2,
        "random_state": 11,
        "center_box": (-7.5, 7.5),
    }
    with open(output, "a") as f:
        f.write(f"Parameters: {params}\n")
        f.write("\n")

    X, y = make_blobs(
        n_samples=params["samples"],
        n_features=params["features"],
        distribution=distribution,
        centers=params["centers"],
        center_box=params["center_box"],
        random_state=params["random_state"],
    )

    # fig = plt.figure(figsize=(15, 12))

    for angle in np.arange(0.0, 100.0, 10.0):
        rotate_and_train(angle)

    # plt.close()
