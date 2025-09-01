import numpy as np
import pandas as pd
import datetime as dt
from scipy.interpolate import interp1d
from scipy.stats import norm
from datetime import datetime, date
import calendar
from typing import Union
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def calculate_theoretical_joint_probs(z_bbb, z_a, rho):
    """
    Calcola la matrice 8×8 delle probabilità teoriche di transizione con soglie z limitate.
    Usa multivariate_normal.cdf e sostituisce ±inf con ±10 per stabilità numerica.
    """
    # Limiti "grandi" al posto di ±inf
    z_bbb = np.clip(z_bbb, -20, 20)
    z_a = np.clip(z_a, -20, 20)

    probs = np.zeros((8, 8))
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    for i in range(8):
        for j in range(8):
            lower = [z_bbb[i], z_a[j]]
            upper = [z_bbb[i+1], z_a[j+1]]

            P11 = multivariate_normal.cdf(upper, mean=mean, cov=cov)
            P10 = multivariate_normal.cdf([upper[0], lower[1]], mean=mean, cov=cov)
            P01 = multivariate_normal.cdf([lower[0], upper[1]], mean=mean, cov=cov)
            P00 = multivariate_normal.cdf(lower, mean=mean, cov=cov)

            probs[i, j] = P11 - P10 - P01 + P00

    return probs


def loss_function(rho, observed, z_bbb, z_a, mode):
    # rho arriva sempre come array di 1 elemento → estraiamolo
    r = float(rho[0])
    theo = calculate_theoretical_joint_probs(z_bbb, z_a, r)
    eps = 1e-10

    if mode == 'MSE':
        return np.sum((theo - observed)**2)
    elif mode == 'MAE':
        return np.sum(np.abs(theo - observed))
    elif mode == 'likelihood':
        mask = observed > 0
        return -np.sum(observed[mask] * np.log(theo[mask] + eps))
    elif mode == 'KL':
        P = observed.flatten()
        Q = np.maximum(theo.flatten(), eps)
        mask = P > 0
        return np.sum(P[mask] * np.log(P[mask] / Q[mask]))
    elif mode == 'JSD':
        P = observed.flatten()
        Q = np.maximum(theo.flatten(), eps)
        M = 0.5 * (P + Q)
        mask = M > 0
        D1 = np.sum(P[mask] * np.log((P[mask] + eps) / (M[mask] + eps)))
        D2 = np.sum(Q[mask] * np.log((Q[mask] + eps) / (M[mask] + eps)))
        return 0.5 * (D1 + D2)
    elif mode == 'weighted MSE':
        i_idx, j_idx = np.meshgrid(np.arange(1,9), np.arange(1,9), indexing='ij')
        W = i_idx + j_idx
        return np.sum(W * (theo - observed)**2)
    elif mode == 'weighted MAE':
        i_idx, j_idx = np.meshgrid(np.arange(1,9), np.arange(1,9), indexing='ij')
        W = i_idx + j_idx
        return np.sum(W * np.abs(theo - observed))
    else:
        raise ValueError(f"Unsupported loss mode: {mode}")

def calibrate_rho(observed, z_bbb, z_a, mode='MSE'):
    """
    Calibra rho minimizzando la loss su observed (8×8), restituisce (rho*, loss*).
    """
    # Vediamo il valore iniziale: magari 0.05
    x0 = np.array([0.05])
    bounds = [(0.0, 1.0)]
    res = minimize(
        fun=lambda x: loss_function(x, observed, z_bbb, z_a, mode),
        x0=x0,
        bounds=bounds,
        tol=1e-10
    )
    return float(res.x[0]), res.fun


def compute_barriers(transition_matrix: pd.DataFrame, rating_row: str) -> np.ndarray:
    """
    Computes asset return thresholds (barriers) corresponding to rating transitions
    for a given initial rating, based on a one-year transition probability matrix.

    Parameters:
        transition_matrix : pd.DataFrame
            Square transition matrix with ratings as both index and columns.
        rating_row : str
            The label of the row representing the initial rating (e.g., "A", "BBB").

    Returns:
        np.ndarray: Array of thresholds (barriers), from Default to AAA.
    """
    # Extract the row of transition probabilities for the given rating
    probs = transition_matrix.loc[rating_row, :].to_numpy()

    # Reverse the order (from Def to AAA)
    cdf = np.cumsum(np.flip(probs))

    # Apply inverse normal (quantile function)
    barriers = norm.ppf(cdf)

    return barriers


def compute_VaR_for_rho(
        rho: float,
        transition_matrix: pd.DataFrame,
        df_1y2ydef: float,
        df_expiry: float,
        bond_mtm_A: float,
        bond_mtm_BBB: float,
        face_value: float,
        recovery_rate: float,
        mc_simulations: int,
        issuers_num_A: int,
        issuers_num_BBB: int,
        rho_A: float,
        rho_BBB: float,
        flag: int,
        seed: int = 0
) -> float:
    """
        Computes the 99.9% Monte Carlo Value at Risk (VaR) for a credit portfolio of zero-coupon bonds,
        using a single-factor Gaussian copula model, including default and migration risk.
    """

    # Initialize random number generator with a fixed seed
    rng = np.random.default_rng(seed=seed)

    # Generate idiosyncratic shocks for each issuer and a common systematic factor
    epsilon_A = rng.normal(size=(mc_simulations, issuers_num_A))
    epsilon_BBB = rng.normal(size=(mc_simulations, issuers_num_BBB))
    y = rng.normal(size=(mc_simulations, 1))

    if flag == 1:
        # Single-factor latent variable model with constant rho
        v_A = np.sqrt(rho) * y + np.sqrt(1 - rho) * epsilon_A
        v_BBB = np.sqrt(rho) * y + np.sqrt(1 - rho) * epsilon_BBB
    elif flag == 2:
        # Single-factor model with rating-dependent rho
        v_A = np.sqrt(rho_A) * y + np.sqrt(1 - rho_A) * epsilon_A
        v_BBB = np.sqrt(rho_BBB) * y + np.sqrt(1 - rho_BBB) * epsilon_BBB


    # Compute default threshold using inverse CDF
    barriers_downgrade_A = compute_barriers(transition_matrix, "A")
    barriers_downgrade_BBB = compute_barriers(transition_matrix, "BBB")

    edges_A = np.concatenate(([-np.inf], barriers_downgrade_A))
    edges_BBB = np.concatenate(([-np.inf], barriers_downgrade_BBB))

    states_A = 9 - np.digitize(v_A, edges_A)
    states_BBB = 9 - np.digitize(v_BBB, edges_BBB)

    # Compute number of defaults and downgrades for issuers with initial rating A
    # Count how many bonds  fall into each rating (1 = AAA, 8 = Default) for each simulation
    # columns ii matrix counts: 1 = AAA, 2 = AA, 3 = A, 4 = BBB, 5 = BB, 6 = B, 7 = CCC, 8 = Default

    counts_A = np.zeros((mc_simulations, 8), dtype=int)
    counts_BBB = np.zeros((mc_simulations, 8), dtype=int)

    for ii in range(1, 9):
        counts_A[:, ii - 1] = np.sum(states_A == ii, axis=1)
        counts_BBB[:, ii - 1] = np.sum(states_BBB == ii, axis=1)

    # Compute average number of downgrades and defaults
    # 1 = AAA, 2 = AA, 3 = A, 4 = BBB, 5 = BB, 6 = B, 7 = CCC, 8 = Default

    avgdowngrade_A = np.mean(counts_A[:,0:8], axis=0)
    avgdowngrade_BBB = np.mean(counts_BBB[:,0:8], axis=0)

    avgdowngrade = np.vstack([avgdowngrade_A, avgdowngrade_BBB])

    # Compute the 0 - 0.5 - year forward discount factor
    frwdis6mesi = df_expiry[1]     # B(0;0,1/2) = B(0,1/2)/B(0,0)

    price1y_A = ((counts_A[:, :7] @ df_1y2ydef.values +
                  counts_A[:, 7] * recovery_rate * frwdis6mesi
                  )/ issuers_num_A) * face_value

    price1y_BBB = ((counts_BBB[:, :7] @ df_1y2ydef.values +
                  counts_BBB[:, 7] * recovery_rate * frwdis6mesi
                  ) / issuers_num_BBB) * face_value

    # Compute loss per scenario
    lossexd_A = bond_mtm_A / df_expiry[2] - price1y_A
    lossexd_BBB = bond_mtm_BBB / df_expiry[2] - price1y_BBB

    # Total portfolio loss per scenario
    lossexd = lossexd_A * issuers_num_A + lossexd_BBB * issuers_num_BBB

    # Compute Value at Risk (VaR) at 99.9% confidence
    confidence_level = 0.999
    VaR_ex = np.percentile(lossexd, confidence_level * 100)

    return VaR_ex, lossexd, avgdowngrade



def datenum(end_date):
    """
        Converts a given date into a numerical representation (Matlab datenum).
        :param end_date: The date to be converted
        :return: The converted date
    """
    # start_date è la settlement date:  2-feb-2023 (datenum 738919)
    start_date = datetime(2023, 2, 2)
    delta=end_date-start_date
    return delta.days+738919

def yearfrac(start_date: date, end_date: date, convention) -> float:
    """
    Replicates MATLAB's yearfrac function for ACT/360, ACT/365, and EU 30/360.
    :param start_date: Start date as a datetime.date object.
    :param end_date: End date as a datetime.date object.
    :param convention: Day count basis (0 = ACT/360, 1 = ACT/365, 2 = EU 30/360).
    :return: Year fraction as a float.
    """
    if convention == 0:  # ACT/360
        return (end_date - start_date).days / 360.0
    elif convention == 1:  # ACT/365
        return (end_date - start_date).days / 365.0
    elif convention == 2:  # EU 30/360
        d1, m1, y1 = start_date.day, start_date.month, start_date.year
        d2, m2, y2 = end_date.day, end_date.month, end_date.year

        # Apply EU 30/360 rules
        d1 = min(d1, 30)
        d2 = 30 if (d1 == 30 and d2 == 31) else min(d2, 30)

        return ((y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)) / 360.0
    else:
        raise ValueError("Unsupported convention")

def zeroRates(dates, discounts):
    """
    Calcola gli zero rates dai fattori di sconto.

    INPUT:
        dates: lista di datetime, scadenze degli strumenti di interesse.
        discounts: lista di fattori di sconto corrispondenti alle date.

    OUTPUT:
        Zero rates annualizzati (%), composti in maniera continua.
    """
    # Data di settlement (hard-coded come nel MATLAB originale)
    settlement = datetime(2023, 2, 2)

    # Calcola le frazioni anno tra la data di settlement e ciascuna data
    delta = np.array([yearfrac(settlement, date, 1) for date in dates[:len(discounts)]])

    # Calcola zero rates (continuamente composti)
    zRates = (-np.log(discounts) / delta) * 100

    return zRates

def business_date_offset(
    base_date: Union[dt.date, pd.Timestamp],
    year_offset: int = 0,
    month_offset: int = 0,
    day_offset: int = 0,
) -> Union[dt.date, pd.Timestamp]:
    """
    Return the closest following business date to a reference date after applying the specified offset.

    Parameters:
        base_date (Union[dt.date, pd.Timestamp]): Reference date.
        year_offset (int): Number of years to add.
        month_offset (int): Number of months to add.
        day_offset (int): Number of days to add.

    Returns:
        Union[dt.date, pd.Timestamp]: Closest following business date to ref_date once the specified
            offset is applied.
    """

    # Adjust the year and month
    total_months = base_date.month + month_offset - 1
    year, month = divmod(total_months, 12)
    year += base_date.year + year_offset
    month += 1

    # Adjust the day and handle invalid days
    day = base_date.day
    try:
        adjusted_date = base_date.replace(
            year=year, month=month, day=day
        ) + dt.timedelta(days=day_offset)
    except ValueError:
        # Set to the last valid day of the adjusted month
        last_day_of_month = calendar.monthrange(year, month)[1]
        adjusted_date = base_date.replace(
            year=year, month=month, day=last_day_of_month
        ) + dt.timedelta(days=day_offset)

    # Adjust to the closest business day
    if adjusted_date.weekday() == 5:  # Saturday
        adjusted_date += dt.timedelta(days=2)
    elif adjusted_date.weekday() == 6:  # Sunday
        adjusted_date += dt.timedelta(days=1)

    return adjusted_date

def getDiscount(dates, discounts, datepags):
    """
    Calcola i fattori di sconto per una serie di date di pagamento (datepags)
    utilizzando i tassi zero derivati dai fattori di sconto degli strumenti.

    Parametri:
        dates: lista di datetime, dove dates[0] è la data di settlement e le altre le scadenze.
        discounts: lista di fattori di sconto corrispondenti alle date.
        datepags: lista (o array) di datetime per cui calcolare il fattore di sconto.

    Ritorna:
        Array numpy dei fattori di sconto per ciascuna data in datepags.
    """
    # La data di settlement è la prima data della lista.
    settlement = dates[0]

    # Calcola il delta in anni (usando ACT/365) per ogni data di pagamento
    delta = np.array([yearfrac(settlement, dp, 1) for dp in datepags])

    # Calcola i tassi zero (continuamente composti, in frazioni) dagli strumenti
    # Utilizza le date degli strumenti a partire dal secondo elemento
    zrates = zeroRates(dates[1:], discounts[1:]) / 100.0

    # Per l'interpolazione, convertiamo le date in frazioni d'anno (rispetto al settlement)
    x_instruments = np.array([yearfrac(settlement, d, 1) for d in dates[1:]])
    x_payments = np.array([yearfrac(settlement, dp, 1) for dp in datepags])

    # Interpolazione lineare dei tassi zero per ottenere il tasso a ciascuna data di pagamento
    f = interp1d(x_instruments, zrates, kind='linear', fill_value="extrapolate")
    interpolated_rates = f(x_payments)

    # Calcola il fattore di sconto per ciascuna data: exp(-tasso * tempo)
    discount_factors = np.exp(-interpolated_rates * delta)

    return discount_factors

