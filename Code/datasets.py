#!/usr/bin/env python3
"""
Generate samples of synthetic data sets.
"""

# Authors: Soham Pal


import numbers
from collections.abc import Iterable
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from sklearn.utils import check_array, check_random_state


def make_blobs(
    n_samples: Union[int, List[int], np.ndarray] = 100,
    n_features: int = 2,
    *,
    distribution: str = "normal",
    centers: Optional[Union[int, np.ndarray]] = None,
    cluster_std: Union[float, List[float], np.ndarray] = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    shuffle: bool = True,
    random_state: Optional[Union[int, np.random.mtrand.RandomState]] = None,
    return_centers: bool = False,
):
    generator = check_random_state(random_state)

    if isinstance(n_samples, numbers.Integral):
        if centers is None:
            centers = 3

        if isinstance(centers, numbers.Integral):
            n_centers = centers
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )
        else:
            centers = check_array(centers)
            n_features = centers.shape[1]
            n_centers = centers.shape[0]
    else:
        n_centers = len(n_samples)
        if centers is None:
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )
        try:
            assert len(centers) == n_centers
        except TypeError as e:
            raise ValueError(
                "Parameter `centers` must be array-like. "
                "Got {!r} instead".format(centers)
            ) from e
        except AssertionError as e:
            raise ValueError(
                f"Length of `n_samples` not consistent with number of "
                f"centers. Got n_samples = {n_samples} and centers = {centers}"
            ) from e
        else:
            centers = check_array(centers)
            n_features = centers.shape[1]

    if hasattr(cluster_std, "__len__") and len(cluster_std) != n_centers:
        raise ValueError(
            f"Length of `cluster_std` not consistent with "
            f"number of centers. Got centers = {centers} "
            f"and cluster_std = {cluster_std}"
        )

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.full(len(centers), cluster_std)

    X = []
    y = []

    if isinstance(n_samples, Iterable):
        n_samples_per_center = n_samples
    else:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers

        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(
            _get_cluster(
                distribution, generator, loc=centers[i], scale=std, size=(n, n_features)
            )
        )
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    if shuffle:
        total_n_samples = np.sum(n_samples)
        indices = np.arange(total_n_samples)
        generator.shuffle(indices)
        X = X[indices]
        y = y[indices]

    if return_centers:
        return X, y, centers
    else:
        return X, y


def _get_cluster(
    distribution: str,
    generator: np.random.RandomState,
    loc: float,
    scale: float,
    size: Tuple[int, int],
):
    if distribution == "gumbel":
        return generator.gumbel(loc=loc, scale=scale, size=size)
    if distribution == "laplace":
        return generator.laplace(loc=loc, scale=scale, size=size)
    if distribution == "logistic":
        return generator.logistic(loc=loc, scale=scale, size=size)
    if distribution == "lognormal":
        return generator.lognormal(mean=loc, sigma=scale, size=size)
    if distribution == "normal" or distribution == "gaussian":
        return generator.normal(loc=loc, scale=scale, size=size)
    if distribution == "vonmises":
        return generator.vonmises(mu=loc, kappa=1.0 / np.square(scale), size=size)
