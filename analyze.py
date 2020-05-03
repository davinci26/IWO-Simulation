#!/usr/bin/env python3

import argparse
import math
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize': (11, 4)})

dataset_path = "./data/"
save_dir = './out/'

hedging_percentage = 100
initial_investment = 5000

subsetSize = 100000


def growth(final, start):
    return (final - start)/start * 100


def cagr(end, start, days):
    return ((end / start)**(365 / days) - 1)*100


def short_shell(dateStart, dateEnd, priceSet, shares_owned, hedging_percentage):
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

    price_end = priceSet.loc[dateEnd].values[0]
    cash_end_required = shares_borrowed * price_end

    profit = cash_at_start - cash_end_required

    # print("Profits/Losses in hedging period {} - {} are {:.2f}".format(dateStart.strftime(
    #     '%d %B %Y'), dateEnd.strftime(
    #     '%d %B %Y'), profit))

    return profit


def simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Line Interface')
    parser.add_argument('-d', '--dataset', type=str, nargs='?')
    parser.add_argument('-v', '--visualize', action='store_true', default=False,
                        help='Visualize data')
    args = parser.parse_args()

    dataset = args.dataset.lower()

    col_list = ["Date", "Close/Last"]

    historicalData = pd.read_csv(
        dataset_path + dataset, usecols=col_list, header=0, parse_dates=[0], index_col=0, skipinitialspace=True)

    print("=================== DATA PREP =============================")
    print(historicalData.head())

    ema_span_low = 5
    ema_span_high = 10
    ema_span_long_term = 120
    reInvesting = True

    subset = historicalData.head(subsetSize)
    subset = subset.sort_index()
    price_at_start = subset.iloc[0].values[0]
    price_at_end = subset.iloc[-1].values[0]

    start_date = subset.iloc[0].name.strftime(
        '%B %Y')
    end_date = subset.iloc[-1].name.strftime(
        '%B %Y')

    fund_days = (subset.iloc[-1].name - subset.iloc[0].name).days

    shares_owned = math.ceil(initial_investment/price_at_start)

    print("=================== INDICES CALCULATION =============================")

    # Calculation of indices
    narrow_ema = subset.ewm(span=ema_span_low, adjust=False).mean()

    wide_ema = subset.ewm(span=ema_span_high, adjust=False).mean()

    long_term_ema = subset.ewm(
        span=ema_span_long_term, adjust=False).mean()

    narrow_shifted = narrow_ema.shift(1)
    wide_shifted = wide_ema.shift(1)

    down_momentum = ((narrow_ema <= wide_ema) &
                     (narrow_shifted >= wide_shifted))

    up_momentum = ((narrow_ema >= wide_ema) & (narrow_shifted <= wide_shifted))

    # Possible index -1 error.
    print(up_momentum.loc[up_momentum['Close/Last'] == True])
    print(down_momentum.loc[down_momentum['Close/Last'] == True])

    # Hedging Strategy Analysis
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

    print("Fund duration: {} days (aprx {:.2f} years)".format(
        fund_days, fund_days / 365))
    print("Initial capital: {}".format(initial_investment))

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
                                   current_shares, hedging_percentage)
        annual_profits[end_year] += hedgeProfits
        if reInvesting:
            price_at_hedge_end = subset.loc[end].values[0]
            # If profits from hedging are positive move again to the position
            if hedgeProfits > 0:
                shares = math.floor(hedgeProfits / price_at_hedge_end)
                current_shares += shares
                cash += hedgeProfits - shares * price_at_hedge_end
            else:
                shares = math.ceil(abs(hedgeProfits) / price_at_hedge_end)
                current_shares -= shares
                residual = shares * price_at_hedge_end - hedgeProfits
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

    if args.visualize:
        # Visualization Layer
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
