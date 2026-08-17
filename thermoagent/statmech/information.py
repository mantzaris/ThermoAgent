"""Information measures used in the statistical-mechanics analysis."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def normalized_probabilities(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError("probabilities must be a one-dimensional vector with at least two states")
    if np.any(probabilities < 0.0) or not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite and nonnegative")
    total = probabilities.sum()
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    return probabilities / total


def shannon_entropy(values: np.ndarray, normalize: bool = True) -> float:
    probabilities = normalized_probabilities(values)
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    if normalize:
        entropy /= np.log(probabilities.size)
    return entropy


def tsallis_entropy(values: np.ndarray, q: float, normalize: bool = True) -> float:
    probabilities = normalized_probabilities(values)
    if abs(q - 1.0) < 1e-7:
        return shannon_entropy(probabilities, normalize=normalize)
    entropy = float((1.0 - np.sum(probabilities ** q)) / (q - 1.0))
    if normalize:
        maximum = (1.0 - probabilities.size ** (1.0 - q)) / (q - 1.0)
        entropy /= maximum
    return entropy


def gini_simpson(values: np.ndarray, normalize: bool = True) -> float:
    probabilities = normalized_probabilities(values)
    impurity = float(1.0 - np.sum(probabilities ** 2))
    if normalize:
        impurity /= 1.0 - 1.0 / probabilities.size
    return impurity


def mutual_information_binary(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("binary samples must have equal one-dimensional shape")
    joint = np.zeros((2, 2), dtype=float)
    for left, right in zip(first, second):
        joint[int(left > 0), int(right > 0)] += 1.0
    joint /= joint.sum()
    marg_left = joint.sum(axis=1)
    marg_right = joint.sum(axis=0)
    information = 0.0
    for i in range(2):
        for j in range(2):
            if joint[i, j] > 0.0:
                information += joint[i, j] * np.log(joint[i, j] / (marg_left[i] * marg_right[j]))
    return float(information)


def macrostate_entropy(samples: np.ndarray, bins: int = 21, q_values: Iterable[float] = (0.5, 1.0, 2.0)) -> Dict[str, float]:
    histogram, _ = np.histogram(np.asarray(samples, dtype=float), bins=bins, range=(-1.0, 1.0))
    output = {"shannon": shannon_entropy(histogram)}
    for q in q_values:
        output["tsallis_q_%s" % str(q).replace(".", "_")] = tsallis_entropy(histogram, q)
    output["gini_simpson"] = gini_simpson(histogram)
    return output
