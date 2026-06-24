#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 12:37:43 2025

@author: eochoa
"""

import numpy as np
import statsmodels.api as sm
#from arch import arch_model
import statsmodels.api as sm


def run_election_night_pred(Y, X, alpha, gamma, tinit=500, split_size=0.75,
                            update_method="Simple", momentum_bw=0.95, random_state=None):
    """
    Adaptive Conformal Inference for election night predictions (Figure 2 in paper).
    """
    rng = np.random.default_rng(random_state)
    T = len(Y)

    alpha_trajectory = np.zeros(T - tinit)
    adapt_err_seq = np.zeros(T - tinit)
    no_adapt_err_seq = np.zeros(T - tinit)
    alphat = alpha

    for t in range(tinit, T):
        # Split data into training and calibration sets
        train_points = rng.choice(t - 1, size=int(split_size * (t - 1)), replace=False)
        cal_points = np.setdiff1d(np.arange(t - 1), train_points)

        X_train, Y_train = X[train_points], Y[train_points]
        X_cal, Y_cal = X[cal_points], Y[cal_points]

        # Add intercept
        X_train_sm = sm.add_constant(X_train)
        X_cal_sm = sm.add_constant(X_cal)

        # Quantile regression fits
        lqrfit_upper = sm.QuantReg(Y_train, X_train_sm).fit(q=1 - alpha / 2)
        lqrfit_lower = sm.QuantReg(Y_train, X_train_sm).fit(q=alpha / 2)

        # Predictions for calibration
        pred_low_cal = lqrfit_lower.predict(X_cal_sm)
        pred_up_cal = lqrfit_upper.predict(X_cal_sm)

        scores = np.maximum(Y_cal - pred_up_cal, pred_low_cal - Y_cal)

        # Predictions for new data point
        x_new = sm.add_constant(X[t:t + 1])
        q_up = lqrfit_upper.predict(x_new)[0]
        q_low = lqrfit_lower.predict(x_new)[0]
        new_score = max(Y[t] - q_up, q_low - Y[t])

        # Compute naive error
        conf_quant_naive = np.quantile(scores, 1 - alpha)
        no_adapt_err_seq[t - tinit] = (conf_quant_naive < new_score).astype(int)

        # Compute adaptive error
        if alphat >= 1:
            adapt_err_seq[t - tinit] = 1
        elif alphat <= 0:
            adapt_err_seq[t - tinit] = 0
        else:
            conf_quant_adapt = np.quantile(scores, 1 - alphat)
            adapt_err_seq[t - tinit] = (conf_quant_adapt < new_score).astype(int)

        # Update alpha_t
        alpha_trajectory[t - tinit] = alphat
        if update_method == "Simple":
            alphat = alphat + gamma * (alpha - adapt_err_seq[t - tinit])
        elif update_method == "Momentum":
            w = momentum_bw ** np.arange(t - tinit + 1)[::-1]
            w /= w.sum()
            alphat = alphat + gamma * (alpha - np.sum(adapt_err_seq[:t - tinit + 1] * w))

        if t % 100 == 0:
            print(f"Done {t} time steps")

    return alpha_trajectory, adapt_err_seq, no_adapt_err_seq


def arima_conformal_forecasting(returns, alpha, gamma, lookback=250,
                                arima_order=(1,0,1), start_up=100, verbose=False,
                                update_method="Simple", momentum_bw=0.95):
    
    """
    Adaptive Conformal Inference for GARCH forecasting (Figure 1 in paper).
    """
    T = len(returns)
    start_up = max(start_up, lookback)
    alphat = alpha

    err_seq_oc = np.zeros(T - start_up + 1)
    err_seq_nc = np.zeros(T - start_up + 1)
    alpha_sequence = np.zeros(T - start_up + 1)
    scores = np.zeros(T - start_up + 1)

    for t in range(start_up, T):
        if verbose:
            print(t)

        # Fit ARIMA model to rolling window
        model = sm.tsa.ARIMA(returns[t - lookback:t], order=arima_order)
        res = model.fit()

        # One-step forecast
        forecast_res = res.get_forecast(steps=1)
        mean_pred = forecast_res.predicted_mean[0]
        sigma_next = np.sqrt(forecast_res.var_pred_mean[0])  # forecast variance proxy

        # Compute conformity score
        scores[t - start_up] = abs(returns[t] ** 2 - sigma_next ** 2) / sigma_next ** 2

        recent_scores = scores[max(t - start_up - lookback + 1, 0):(t - start_up)]

        # Errors
        if len(recent_scores) > 0:
            err_seq_oc[t - start_up] = int(scores[t - start_up] > np.quantile(recent_scores, 1 - alphat))
            err_seq_nc[t - start_up] = int(scores[t - start_up] > np.quantile(recent_scores, 1 - alpha))

        # Update alpha_t
        alpha_sequence[t - start_up] = alphat
        if update_method == "Simple":
            alphat = alphat + gamma * (alpha - err_seq_oc[t - start_up])
        elif update_method == "Momentum":
            w = momentum_bw ** np.arange(t - start_up + 1)[::-1]
            w /= w.sum()
            alphat = alphat + gamma * (alpha - np.sum(err_seq_oc[:t - start_up + 1] * w))

        if t % 100 == 0:
            print(f"Done {t} steps")

    return alpha_sequence, err_seq_oc, err_seq_nc


