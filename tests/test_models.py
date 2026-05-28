from __future__ import annotations

from soxx_mvp.models import fit_logistic, fit_ridge


def test_sklearn_model_wrappers_are_deterministic() -> None:
    x_rows = [
        [0.0, 0.1],
        [1.0, 1.1],
        [2.0, 1.9],
        [3.0, 3.2],
    ]
    y_direction = [0, 0, 1, 1]
    y_return = [-0.01, -0.005, 0.01, 0.02]

    logistic_a = fit_logistic(x_rows, y_direction, l2=0.1, max_iter=200)
    logistic_b = fit_logistic(x_rows, y_direction, l2=0.1, max_iter=200)
    ridge_a = fit_ridge(x_rows, y_return, l2=0.1)
    ridge_b = fit_ridge(x_rows, y_return, l2=0.1)

    x_current = [1.5, 1.4]
    assert logistic_a.predict_proba_one(x_current) == logistic_b.predict_proba_one(x_current)
    assert ridge_a.predict_one(x_current) == ridge_b.predict_one(x_current)


def test_single_class_logistic_uses_constant_probability() -> None:
    model = fit_logistic([[0.0], [1.0], [2.0]], [1, 1, 1], l2=0.1, max_iter=100)
    assert model.predict_proba_one([99.0]) == 1.0
