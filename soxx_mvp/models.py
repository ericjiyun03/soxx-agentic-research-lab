from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ConstantLogisticModel:
    probability: float

    def predict_proba_one(self, x_row: list[float]) -> float:
        return self.probability


@dataclass
class SklearnLogisticModel:
    pipeline: Pipeline

    def predict_proba_one(self, x_row: list[float]) -> float:
        probabilities = self.pipeline.predict_proba(np.asarray([x_row], dtype=float))[0]
        classes = list(self.pipeline.named_steps["model"].classes_)
        if 1 not in classes:
            return 0.0
        return float(probabilities[classes.index(1)])


@dataclass
class SklearnRidgeModel:
    pipeline: Pipeline

    def predict_one(self, x_row: list[float]) -> float:
        return float(self.pipeline.predict(np.asarray([x_row], dtype=float))[0])


def fit_logistic(
    x_rows: list[list[float]],
    y_values: list[int],
    *,
    epochs: int | None = None,
    learning_rate: float | None = None,
    l2: float | None = None,
    max_iter: int | None = None,
    c: float | None = None,
    **_: object,
) -> ConstantLogisticModel | SklearnLogisticModel:
    del learning_rate
    if not x_rows:
        raise ValueError("Cannot fit logistic model on empty rows")

    unique_classes = set(int(value) for value in y_values)
    if len(unique_classes) == 1:
        return ConstantLogisticModel(probability=float(next(iter(unique_classes))))

    regularization = float(l2 if l2 is not None else 1.0)
    inverse_regularization = float(c if c is not None else (1.0 / regularization if regularization > 0 else 1e6))
    iterations = int(max_iter if max_iter is not None else max(100, int(epochs or 50) * 10))

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=inverse_regularization,
                    max_iter=iterations,
                    solver="lbfgs",
                    random_state=None,
                ),
            ),
        ]
    )
    pipeline.fit(np.asarray(x_rows, dtype=float), np.asarray(y_values, dtype=int))
    return SklearnLogisticModel(pipeline=pipeline)


def fit_ridge(
    x_rows: list[list[float]],
    y_values: list[float],
    *,
    epochs: int | None = None,
    learning_rate: float | None = None,
    l2: float | None = None,
    alpha: float | None = None,
    **_: object,
) -> SklearnRidgeModel:
    del epochs, learning_rate
    if not x_rows:
        raise ValueError("Cannot fit ridge model on empty rows")

    regularization = float(alpha if alpha is not None else (l2 if l2 is not None else 1.0))
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=regularization)),
        ]
    )
    pipeline.fit(np.asarray(x_rows, dtype=float), np.asarray(y_values, dtype=float))
    return SklearnRidgeModel(pipeline=pipeline)
