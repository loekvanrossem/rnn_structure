import numpy as np
from numba import njit
import scipy

import torch


@njit
def der(t, z, eta_h, eta_y, dx2, dy2):
    """Right hand side of the 3d ode system."""
    h, y, w = z
    return [
        -eta_h * dx2 * w,
        -w * (eta_y * h + eta_h * dx2 * y / h),
        -(1 / 2) * eta_y * (3 * w - y + dy2) * h
        - (1 / 2) * eta_h * dx2 * (y + w) * w / h,
    ]


@njit
def loss(t, z, eta_y_mean, dy2, y0_mean):
    """Mean squared loss for two datapoints."""
    L = (1 / 4) * (
        2 * np.exp(-2 * t * eta_y_mean) * y0_mean + z[2] + (1 / 2) * (dy2 - z[1])
    )
    return L


def rep_sim(h0, y0, w0, dx2, dy2):
    """Return the representational distance of the two point system."""
    eta_h = 0.0001
    eta_y = eta_h * 1

    t_max = 100000

    sol = scipy.integrate.solve_ivp(
        der,
        [0, t_max],
        [h0, y0, w0],
        args=(eta_h, eta_y, dx2, dy2),
        dense_output=False,
    )

    z = sol.y[:, -1]
    return z[0]


def optimize_eta(h2, y2, w, dx2, dy2, guesses=np.logspace(-6, 2, 200)):
    """Find the best fitting values for eta_h and eta_y."""
    h0, y0, w0 = h2[0], y2[0], w[0]
    n_epochs = len(h2)

    ratio = h2[-1] ** 2 / (dx2 * dy2)

    def model_accuracy(pars):
        eta_h, eta_y = pars

        t_max = n_epochs

        sol = scipy.integrate.solve_ivp(
            der,
            [0, t_max],
            [h0, y0, w0],
            args=(eta_h, eta_y, dx2, dy2),
            dense_output=True,
            method="Radau",
        )

        t = np.linspace(0, t_max, n_epochs)
        z = sol.sol(t)

        loss = (
            np.sum((h2 - z[0]) ** 2)
            + np.sum((y2 - z[1]) ** 2)
            + np.sum((w - z[2]) ** 2)
        )
        return loss

    # Find optimal etas
    guess = guesses[
        np.argmin([model_accuracy(guess * np.array([ratio, 1])) for guess in guesses])
    ]
    optimal = scipy.optimize.minimize(model_accuracy, guess * np.array([ratio, 1]))
    loss = optimal.fun
    eta_h_opt, eta_y_opt = optimal.x

    print(f"Loss: {loss}")

    return (eta_h_opt, eta_y_opt, loss)


def optimize_eta_y_mean(z, train_loss, dy2, y0_mean):
    """Find the best fitting eta_y_mean."""
    n_epochs = len(train_loss)

    def model_accuracy(pars):
        (eta_y_mean,) = pars

        t_max = n_epochs
        t = np.linspace(0, t_max, n_epochs)

        pred = loss(t, z, eta_y_mean, dy2, y0_mean)

        model_loss = np.sum((pred - train_loss) ** 2)
        return model_loss

    # Optimize eta
    guesses = np.logspace(-5, 0, 10)
    guess = guesses[np.argmin([model_accuracy((guess,)) for guess in guesses])]

    optimal = scipy.optimize.minimize(model_accuracy, (guess,))
    print(f"Loss: {optimal.fun}")

    (eta_y_mean_opt,) = optimal.x

    return eta_y_mean_opt
