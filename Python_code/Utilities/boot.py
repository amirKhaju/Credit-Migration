import calendar
import datetime as dt
from datetime import datetime, date
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d


class DayCount(Enum):
    EU_30_360 = 2  # EU 30/360
    ACT_360 = 0    # ACT/360
    ACT_365 = 1    # ACT/365

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

def next_business_day(date):
    """ Returns the next business day."""
    next_day = np.busday_offset(date, 0, roll='forward')
    return next_day

def adjust_non_business_days(dates):
    """
        Adjust dates to ensure they are business days.
        This function checks if each date in the given list is a business day.
        If a date falls on a non-business day, it is
        adjusted to the next available business day.
        :param dates: The date to be adjusted
        :return: adjusted_dates_conv: list of datetime.datetime
    """
    # convert in datetime64[D] format
    adjusted_dates = np.array(dates, dtype='datetime64[D]')
    for i in range(len(adjusted_dates)):
        if not np.is_busday(adjusted_dates[i]):
            adjusted_dates[i] = next_business_day(adjusted_dates[i])

    # convert adjusted_dates in datetime.data from datetime64[D]
    adjusted_dates = adjusted_dates.astype('O')

    # convert datesswap in datetime.datetime format from  datetime.data  in the variable dateswapconv
    adjusted_dates_conv = [datetime(d.year, d.month, d.day) for d in adjusted_dates]

    return adjusted_dates_conv

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

def zeroRatesi(dates, discounts,i):

    delta = yearfrac(dates[0], dates[i], DayCount.ACT_365.value)
    zRates = (-np.log(discounts) / delta) * 100.0
    return zRates

def swaperate(datesSet, ratesSet):
    """
    This function computes swap rates by extracting market swap data, adjusting
    payment dates to business days, and performing interpolation to generate
    swap rates up to a specified final date.
    :param datesSet:  a dictionary containing dates.
    :param ratesSet:  a dictionary containing rates
    :return:    datesswap: A list of adjusted swap dates
    :return:    rateswap: An array of interpolated swap rates corresponding to the computed dates
    """
    # Calculate the mid-swap rates
    ratemid = (ratesSet['swaps'][:, 0] + ratesSet['swaps'][:, 1]) / 2
    # Assign the first 11 swap dates from the dataset
    datesswap=[]
    datesswap[0: 11]=datesSet['swaps'][0:11]

    settlements = datetime(2035, 2, 2)
    finaldate = datetime(2073, 2, 2)
    years_diff = (finaldate - settlements).days // 365
    # Generate an array of years from 1 to years_diff
    calyears = np.arange(1, years_diff + 1)

    # Create a list of dates by adding the calculated years to the settlement date
    ipodate = [settlements + relativedelta(years=int(y)) for y in calyears]
    # Assing the new dates to the dates swap list
    datesswap[11:50]=ipodate

    # Adjust non-business days
    datesswap=adjust_non_business_days(datesswap)

    # convert datesswap in datenum in the variable dateswap_datenum
    dateswap_datenum = [datenum(date) for date in datesswap]

    # convert datesSet['swaps'] in datenum in the variable known_dates
    known_dates = np.array([datenum(date) for date in datesSet['swaps']])

    # Perform cubic spline interpolation to estimate missing swap rates
    funct=CubicSpline(known_dates,ratemid, axis=0, bc_type='not-a-knot', extrapolate=True)
    rateswap=funct(dateswap_datenum)

    return datesswap, rateswap



def bootstrap(datesSet, ratesSet):
    """
    This function computes the discount factors using the bootstrap method.
    The process involves extracting market rates for deposits, futures, and swaps,
    then using them to calculate discount factors iteratively.
    :param datesSet: Dictionary containing settlement, deposit, futures, and swap dates.
    :param ratesSet: Dictionary containing deposit, futures, and swap rates.
    :return: List of dates corresponding to calculated discount factors.
    :return: List of discount factors computed using bootstrap.
    """
    n_total = 61    # Total number of dates in the curve
    dates = [None] * n_total  # Initialize dates list
    discounts = np.zeros(n_total)  # Initialize discounts array

    # Set the settlement date
    dates[0] = datesSet['settlement']
    start_date = dates[0]
    discounts[0] = 1  # Discount to 1 for settlement date

    # Compute mid rates for deposits and futures by averaging bid and ask rates
    rate_mid_depos = (ratesSet['depos'][:, 0] + ratesSet['depos'][:, 1]) / 2
    rate_mid_futures = (ratesSet['futures'][:, 0] + ratesSet['futures'][:, 1]) / 2

    # Populate deposit dates
    dates[1:5] = datesSet['depos'][0:4]

    # Populate Futures Dates
    for i in range(0,7):
        dates[i+5] = datesSet['futures'][i][1]

    # Compute Discount Factors for deposits
    for i in range(1, 5):
        discounts[i] = 1 / (1 + (yearfrac(dates[0], dates[i], DayCount.ACT_360.value) * rate_mid_depos[i - 1]))

    # Compute Zero Rates for Futures
    ratezz = np.array([zeroRatesi(dates, discounts[i + 3], i + 3) / 100 for i in range(2)])
    datezz = np.array([datenum(dates[3]), datenum(dates[4])])

    # Interpolate for the first future
    zerorateinterpl=np.interp(datenum(datesSet['futures'][0][0]),datezz, ratezz)

    # Compute Discount Factor for the First Future
    discountsfut=np.exp(-yearfrac(dates[0],datesSet['futures'][0][0],DayCount.ACT_365.value)*zerorateinterpl)
    B_ti_tii = 1 / (1 + rate_mid_futures[0] * yearfrac(datesSet['futures'][0][0], datesSet['futures'][0][1], DayCount.ACT_360.value))
    discounts[5] = discountsfut* B_ti_tii

    # Compute discount factors for subsequent futures contracts
    for i in range(1,7):
        B_ti_tii= 1 / (1 + rate_mid_futures[i] * yearfrac(datesSet['futures'][i][0], datesSet['futures'][i][1], DayCount.ACT_360.value))
        prev_date = datesSet['futures'][i - 1][1]
        curr_date = datesSet['futures'][i][0]

        if curr_date == prev_date:
            discounts[i + 5] = discounts[i + 4] * B_ti_tii
        elif curr_date > prev_date:
            # If the settlement date of the current future is greater than the expiry date of the previous one,
            # use the formula to compute the discount factor based on the interpolated zero rate.

            zerorate = zeroRatesi(dates, discounts[i + 4],i+4) / 100
            B_t0_ti = np.exp(-yearfrac(dates[0], datesSet['futures'][i][0], DayCount.ACT_365.value) * zerorate)
            discounts[i + 5] = B_t0_ti * B_ti_tii
        else:
            # If the current future's start date is before the previous one, perform interpolation

            zerorate = np.zeros(2)
            zerorate[0] = zeroRatesi(dates, discounts[i+3],i+3) / 100
            zerorate[1] = zeroRatesi(dates, discounts[i+4], i + 4) / 100
            datezz[0] = datenum(dates[i+3])
            datezz[1] = datenum(dates[i+4])
            zerorateinterpl = np.interp(datenum(curr_date), datezz, zerorate)
            B_t0_ti = np.exp(-yearfrac(dates[0], datesSet['futures'][i][0], DayCount.ACT_365.value) * zerorateinterpl)
            discounts[i + 5] = B_t0_ti * B_ti_tii

    # Compute Discount Factors for Swaps
    dateswap1yd = datetime(2024, 2, 2)
    dateswap1y = datenum(dateswap1yd)

    # Compute zero rates for swaps
    for i in range(0,2):
        zerorate[i] = zeroRatesi(dates, discounts[i+7],i+7) / 100

    # Interpolate zero rate for the one-year swap date
    datezz = [datenum(dates[7]), datenum(dates[8])]
    rateswaptemp = np.interp( dateswap1y,datezz, zerorate)

    # Obtain swap dates and rates
    [datesswap, rateswaps] = swaperate(datesSet, ratesSet)
    dates[12: 13 + len(datesswap) - 1]=datesswap

    b = np.zeros(50)
    b[0] = np.exp(-yearfrac(datesSet['settlement'], dateswap1yd, DayCount.ACT_365.value) * rateswaptemp)

    # Construct a list of relevant dates for swaps
    dateutili = [dateswap1yd] + datesswap

    sum=0
    sum = yearfrac(datesSet['settlement'], dateutili[0], DayCount.EU_30_360.value) * b[0]

    for i in range (0,49):
        b[i + 1] = (1 - rateswaps[i] * sum) / (1 + yearfrac(dateutili[i], dateutili[i + 1], DayCount.EU_30_360.value) * rateswaps[i])
        sum= sum + yearfrac(dateutili[i], dateutili[i + 1], DayCount.EU_30_360.value) * b[i + 1]

    discounts[12: len(datesswap) + 13 - 1] = b[1:]

    return dates, discounts



def zeroRates(dates, discounts):

    settlement = datetime(2023, 2, 2)
    delta = np.array([yearfrac(settlement, date, 1) for date in dates[:len(discounts)]])
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
    delta = np.array([yearfrac(settlement, dp, 1) for dp in datepags])
    zrates = zeroRates(dates[1:], discounts[1:]) / 100.0

    x_instruments = np.array([yearfrac(settlement, d, 1) for d in dates[1:]])
    x_payments = np.array([yearfrac(settlement, dp, 1) for dp in datepags])

    f = interp1d(x_instruments, zrates, kind='linear', fill_value="extrapolate")
    interpolated_rates = f(x_payments)

    discount_factors = np.exp(-interpolated_rates * delta)

    return discount_factors

