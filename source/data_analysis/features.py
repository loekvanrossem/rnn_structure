from typing import Callable
from tqdm import trange, tqdm
from multiprocessing import Pool

import random
from numba import njit, jit
import numpy as np


@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@njit
def distxy_opt(image_1: np.ndarray, image_2: np.ndarray, label_1: int, label_2: int):
    if label_1 == label_2:
        return float(0)
    d = image_1 - image_2
    return np.sqrt(np.sum(d**2))


@njit
def distyy_opt(image_1: np.ndarray, image_2: np.ndarray, label_1: int, label_2: int):
    if label_1 == label_2:
        return float(0)
    return float(1)


def feature_strength(classes: list[list], metric: Callable):
    """
    Returns the feature strength of a dataset partition
    """
    dist = 0
    for n_1, class_1 in enumerate(classes):
        for class_2 in classes[n_1 + 1 :]:
            for point_1 in tqdm(class_1):
                for point_2 in class_2:
                    dist += metric(point_1, point_2)
    return dist


def dist_split(dataset: list, metric: Callable, n_iter: int = 1000, n_sample=1000):
    """
    Find maximal distance partition of dataset.

    dataset : list
        List of all datapoints
    metric : Callable
        Function on two datapoints return a float
    n_iter : int, optional
        Number of iterations
    n_sample : int, optional
        Maximum number of samples to compute distance between classes
    """
    class_A = dataset
    class_B = []
    N = len(dataset)
    iterator = trange(n_iter)
    for n in iterator:
        # Select datapoint
        index = np.random.randint(N)
        try:
            point = class_A.pop(index)
        except IndexError:
            point = class_B.pop((N - 1) - index)

        # Compute distance change
        if len(class_A) > n_sample:
            A = random.sample(class_A, n_sample)
        else:
            A = class_A
        if len(class_B) > n_sample:
            B = random.sample(class_B, n_sample)
        else:
            B = class_B

        dist_A = np.sum([metric(point, a) for a in A])
        dist_B = np.sum([metric(point, b) for b in B])

        # Choose class
        prob_A = sigmoid(dist_B - dist_A)
        if np.random.random() < prob_A:
            class_A.append(point)
        else:
            class_B.append(point)

        # Print progress
        if n % 100 == 0:
            iterator.set_description(
                f"Splitting... (#A: {len(class_A)}, #B: {len(class_B)})"
            )

    return class_A, class_B
