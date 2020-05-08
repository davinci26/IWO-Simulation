#!/usr/bin/env python3

from dataclasses import dataclass
import os.path
import argparse
import math
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize': (11, 4)})

hedging_percentage = 100
initial_investment = 5000
reInvesting = True
interest_rate = 0.15

subsetSize = None


def growth(final, start):
    return (final - start)/start * 100


def cagr(end, start, days):
    return ((end / start)**(365 / days) - 1)*100


def simulate(ema_span_low, ema_span_high, ema_span_long_term, subset, fund_days, verbose=False):

    price_at_start = subset.iloc[0].values[0]
    price_at_end = subset.iloc[-1].values[0]
    shares_owned = math.ceil(initial_investment/price_at_start)

    # Calculation of indices
    narrow_ema = subset.ewm(span=ema_span_low, adjust=False).mean()

    wide_ema = subset.ewm(span=ema_span_high, adjust=False).mean()

    long_term_ema = subset.ewm(
        span=ema_span_long_term, adjust=False).mean()

    # Hedging Strategy Analysis
    if verbose:
        print("=================== HEDGING ANALYSIS ============================")
    """
    You are hedged when:
        1. narrow EMA < wide EMA
        2. price < long_term_ema
    """
    hedged = (subset < long_term_ema) & (narrow_ema <= wide_ema)
    hedged_shifted_r = hedged.shift(1)
    hedged_shifted_l = hedged.shift(-1)

    hedge_start = hedged_shifted_l & (~hedged)
    hedge_start = hedge_start.loc[hedge_start['Close/Last'] == True]
    hedge_end = hedged_shifted_r & (~hedged)
    hedge_end = hedge_end.loc[hedge_end['Close/Last'] == True]

    # TODO: There might be an awkard case where the do the backtesting while
    # we already hedged. In that case this might break since the dataframes
    # do not have the same shape.
    if verbose:
        print("Fund duration: {} days (aprx {:.2f} years)".format(
            fund_days, fund_days / 365))
        print("Initial capital: {}".format(initial_investment))

        print("Hyper-parameters: (Narrow window, Wide window, Long Term window) ({},{},{}))".format(
            ema_span_low, ema_span_high, ema_span_long_term))
        if reInvesting:
            print("Profits/losses made from hedging will be re-invested in the fund")
        else:
            print("Profits/losses made will be kept as cash")

    cash = 0
    current_shares = shares_owned
    annual_profits = defaultdict(int)
    for i in range(0, min(hedge_start.shape[0], hedge_end.shape[0])):
        start = hedge_start.iloc[i].name
        end = hedge_end.iloc[i].name
        end_year = end.strftime('%Y')
        hedgeProfits = short_shell(start, end, subset,
                                   current_shares, hedging_percentage, interest_rate)
        annual_profits[end_year] += hedgeProfits
        if reInvesting:
            price_at_hedge_end = subset.loc[end].values[0]
            # If profits from hedging are positive move again to the position
            if hedgeProfits > 0:
                cash += hedgeProfits
                shares = math.floor(cash / price_at_hedge_end)
                current_shares += shares
                cash -= shares * price_at_hedge_end
            else:
                shares = math.ceil(abs(hedgeProfits) / price_at_hedge_end)
                current_shares -= shares
                residual = shares * price_at_hedge_end - hedgeProfits
                cash += residual
                if current_shares < 0:
                    raise ValueError(
                        "Simulation lead to bankcrupcy at hedge {} - {}".format(start, end))
        else:
            cash += hedgeProfits

    hold_value = price_at_end * shares_owned
    hold_cagr = cagr(hold_value, initial_investment, fund_days)
    hedged_portofolio_value = price_at_end * current_shares
    hedged_terminal_value = hedged_portofolio_value + cash
    hedged_cagr = cagr(hedged_terminal_value, initial_investment, fund_days)

    if verbose:
        print("Results Overview:")
        print("-> Terminal value with Long & Hold {:.2f} and a CAGR of {:.2f}%".format(
            hold_value, hold_cagr))
        print("-> Terminal value with Hedges {:.2f} (Portofolio: {:.2f} Cash: {:.2f}) and a CAGR of {:.2f}%".format(
            hedged_terminal_value, hedged_portofolio_value, cash, hedged_cagr))

        if hedged_cagr > hold_cagr:
            print(
                "-> Hedging outperforms Long & Hold by {:.2f}x".format(hedged_terminal_value/hold_value))
        else:
            print(
                "-> Long & Hold outperforms hedging by {:.2f}x".format(hold_value/hedged_terminal_value))
        print("Hedging Cost Breakdown per Year")
        for key in annual_profits:
            print("-> Year: {} - Profit/Loss: {:.2f} ".format(key,
                                                              annual_profits[key]))
        print("=================================================================")

    return hedged_cagr, hedged_portofolio_value, cash, hold_value, hold_cagr, annual_profits,  narrow_ema, wide_ema, long_term_ema, hedged


def short_shell(dateStart, dateEnd, priceSet, shares_owned, hedging_percentage, interest_rate):
    """
    This means that we borrow the minumum number of shares n such that n * price_start >= amount
    """
    # TODO: This probably need to adjusted to also calculate the maximum paper loss
    # since this might trigger margin calls.

    price_start = priceSet.loc[dateStart].values[0]

    portofolio_valuation = shares_owned * price_start
    portofolio_hedged = portofolio_valuation * hedging_percentage / 100
    shares_borrowed = math.ceil(portofolio_hedged / price_start)

    cash_at_start = shares_borrowed * price_start

    interest = (cash_at_start * interest_rate / 360) * \
        (max((dateEnd - dateStart).days, 1))

    price_end = priceSet.loc[dateEnd].values[0]
    cash_end_required = shares_borrowed * price_end
    cash_end_required += interest

    profit = cash_at_start - cash_end_required

    return profit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Line Interface')
    parser.add_argument('-d', '--dataset', type=str,
                        nargs='?', help='Path to the dataset')
    parser.add_argument('-v', '--visualize', action='store_true', default=False,
                        help='Visualize data')
    parser.add_argument('-p', '--optimize', action='store_true', default=False,
                        help='Optimize the hyper-parameters of the simulation')
    args = parser.parse_args()

    dataset_path = "./data/"

    col_list = ["Date", "Close/Last"]
    if not args.dataset or not os.path.exists(args.dataset.lower()):
        print("ERROR: The path {} does not seem to exist".format(args.dataset))
        print("You can download the data from https://www.nasdaq.com/market-activity/funds-and-etfs/iwo/historical")
        exit(1)

    dataset = args.dataset.lower()
    try:
        historicalData = pd.read_csv(
            dataset, usecols=col_list, header=0, parse_dates=[0], index_col=0, skipinitialspace=True)
    except:
        print("ERROR: Failed to load {} as CSV".format(dataset))
        print("Make sure it follows the format (Date, Close/Last, Volume, Open, High, Low)")
        print("You can download the data from https://www.nasdaq.com/market-activity/funds-and-etfs/iwo/historical")
        exit(1)

    print("=================== DATA PREP =============================")
    print(historicalData.head())

    if subsetSize is not None:
        subset = historicalData.head(subsetSize)
    else:
        subset = historicalData

    subset = subset.sort_index()

    ema_span_low_space = [3, 4, 5, 6, 7, 8, 9, 10]
    ema_span_high_space = [x + 5 for x in ema_span_low_space]
    ema_span_long_term_space = [105, 110, 115, 120, 125, 130, 135]

    fund_days = (subset.iloc[-1].name - subset.iloc[0].name).days

    maxL = 0
    hyperParameters = None
    performance = None
    if args.optimize:
        for ema_span_long_term in ema_span_long_term_space:
            for ema_span_high in ema_span_high_space:
                for ema_span_low in ema_span_low_space:
                    results = simulate(ema_span_low, ema_span_high,
                                       ema_span_long_term, subset, fund_days)
                    if results[0] > maxL:
                        maxL = results[0]
                        performance = results
                        hyperParameters = (
                            ema_span_low, ema_span_high, ema_span_long_term)

        hedged_cagr, hedged_portofolio_value, cash, hold_value, hold_cagr, annual_profits,  narrow_ema, wide_ema, long_term_ema, hedged = performance
        hedged_terminal_value = hedged_portofolio_value + cash
        print("Simulation Settings")
        print("Fund duration: {} days (aprx {:.2f} years)".format(
            fund_days, fund_days / 365))
        print("Initial capital: {}".format(initial_investment))

        print("Optimal Hyper-Parameters")
        print("-> Hyper-parameters: (Narrow window, Wide window, Long Term window) ({},{},{}))".format(
            ema_span_low, ema_span_high, ema_span_long_term))
        if reInvesting:
            print("Profits/losses made from hedging will be re-invested in the fund")
        else:
            print("Profits/losses made will be kept as cash")
        print("Results Overview:")
        print("-> Terminal value with Long & Hold {:.2f} and a CAGR of {:.2f}%".format(
            hold_value, hold_cagr))
        print("-> Terminal value with Hedges {:.2f} (Portofolio: {:.2f} Cash: {:.2f}) and a CAGR of {:.2f}%".format(
            hedged_terminal_value, hedged_portofolio_value, cash, hedged_cagr))

        if hedged_cagr > hold_cagr:
            print(
                "-> Hedging outperforms Long & Hold by {:.2f}x".format(hedged_terminal_value/hold_value))
        else:
            print(
                "-> Long & Hold outperforms hedging by {:.2f}x".format(hold_value/hedged_terminal_value))
        print("Hedging Cost Breakdown per Year")
        for key in annual_profits:
            print("-> Year: {} - Profit/Loss: {:.2f} ".format(key,
                                                              annual_profits[key]))
    else:
        ema_span_low = 5
        ema_span_high = 10
        ema_span_long_term = 120
        performance = simulate(ema_span_low, ema_span_high,
                               ema_span_long_term, subset, fund_days, verbose=True)

    if args.visualize:
        hedged_cagr, hedged_portofolio_value, cash, hold_value, hold_cagr, annual_profits,  narrow_ema, wide_ema, long_term_ema, hedged = performance
        # Visualization Layer
        start_date = subset.iloc[0].name.strftime(
            '%B %Y')
        end_date = subset.iloc[-1].name.strftime(
            '%B %Y')

        ax = subset.rename(columns={"Close/Last": "IWO Price"}).plot(lw=1,
                                                                     figsize=(14, 7), label='IWO Technical Analysis')

        narrow_ema.rename(columns={"Close/Last": "EMA (window {})".format(ema_span_low)}).plot(
            ax=ax, lw=2)

        wide_ema.rename(columns={"Close/Last": "EMA (window {})".format(ema_span_high)}).plot(
            ax=ax, lw=2)

        long_term_ema.rename(columns={"Close/Last": "EMA (window {})".format(ema_span_long_term)}).plot.area(
            ax=ax, lw=2,  alpha=0.1)

        plt.title('IWO Close Price from {} to {}'.format(
            start_date, end_date), fontsize=16)
        plt.tick_params(labelsize=12)

        ax.fill_between(
            hedged.index, 0, 1, where=hedged['Close/Last'], transform=ax.get_xaxis_transform(), alpha=0.3, label="Hedges On")

        plt.legend(loc='upper left', fontsize=12)
        plt.show()
