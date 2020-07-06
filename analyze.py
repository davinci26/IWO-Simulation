#!/usr/bin/env python3

from utility import cagr, growth, buy
import logging
from dataclasses import dataclass
import os.path
import argparse
import math
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize': (11, 4)})

initial_investment = 5000
price_column = 'Close'


logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


class Ema_crossover_parameters:
    def __init__(self, ema_span_low=5, ema_span_high=10, ema_span_long_term=105):
        self.ema_span_low = ema_span_low
        self.ema_span_high = ema_span_high
        self.ema_span_long_term = ema_span_long_term


def ema_crossover_signal(long_etf, parameters):
    ema_span_low = parameters.ema_span_low
    ema_span_high = parameters.ema_span_high
    ema_span_long_term = parameters.ema_span_long_term

    # Calculation of indices
    narrow_ema = long_etf.ewm(span=ema_span_low, adjust=False).mean()

    wide_ema = long_etf.ewm(span=ema_span_high, adjust=False).mean()

    long_term_ema = long_etf.ewm(
        span=ema_span_long_term, adjust=False).mean()

    # Hedging Strategy Analysis
    logger.info("HEDGING ANALYSIS")

    """
    You are hedged when:
        1. narrow EMA < wide EMA
        2. price < long_term_ema
    """
    hedged = (long_etf < long_term_ema) & (narrow_ema <= wide_ema)
    hedged_shifted_r = hedged.shift(1)
    hedged_shifted_l = hedged.shift(-1)

    hedge_start = hedged_shifted_l & (~hedged)
    hedge_start = hedge_start.loc[hedge_start[price_column] == True]
    hedge_end = hedged_shifted_r & (~hedged)
    hedge_end = hedge_end.loc[hedge_end[price_column] == True]

    return hedge_start, hedge_end


def visualize_signal(etf, parameters, etf_name="IWO"):

    ema_span_low = parameters.ema_span_low
    ema_span_high = parameters.ema_span_high
    ema_span_long_term = parameters.ema_span_long_term

    narrow_ema = long_etf.ewm(span=ema_span_low, adjust=False).mean()
    wide_ema = long_etf.ewm(span=ema_span_high, adjust=False).mean()

    long_term_ema = long_etf.ewm(
        span=ema_span_long_term, adjust=False).mean()

    hedged = (long_etf < long_term_ema) & (narrow_ema <= wide_ema)
    hedged_shifted_r = hedged.shift(1)
    hedged_shifted_l = hedged.shift(-1)

    hedge_start = hedged_shifted_l & (~hedged)
    hedge_start = hedge_start.loc[hedge_start[price_column] == True]
    hedge_end = hedged_shifted_r & (~hedged)
    hedge_end = hedge_end.loc[hedge_end[price_column] == True]

    # Visualization Layer
    start_date = etf.iloc[0].name.strftime(
        '%B %Y')
    end_date = etf.iloc[-1].name.strftime(
        '%B %Y')

    ax = etf.rename(columns={price_column: "{} Price".format(etf_name)}).plot(lw=1,
                                                                              figsize=(14, 7), label='{} Technical Analysis'.format(etf_name))

    narrow_ema.rename(columns={price_column: "EMA (window {})".format(ema_span_low)}).plot(
        ax=ax, lw=2)

    wide_ema.rename(columns={price_column: "EMA (window {})".format(ema_span_high)}).plot(
        ax=ax, lw=2)

    long_term_ema.rename(columns={price_column: "EMA (window {})".format(ema_span_long_term)}).plot.area(
        ax=ax, lw=2,  alpha=0.1)

    plt.title('{} Close Price from {} to {}'.format(etf_name,
                                                    start_date, end_date), fontsize=16)
    plt.tick_params(labelsize=12)

    ax.fill_between(
        hedged.index, 0, 1, where=hedged[price_column], transform=ax.get_xaxis_transform(), alpha=0.3, label="Hedges Required")

    plt.legend(loc='upper left', fontsize=12)
    plt.show()


class Short_sell_paremeters:
    def __init__(self, interest_rate=0.15, hedging_percentage=100):
        self.interest_rate = interest_rate
        self.hedging_percentage = hedging_percentage


def short_sell(dateStart, dateEnd, priceSet, shares_owned, hedging_percentage, interest_rate):
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


def short_sell_strategy(long_etf, start, end, cash, current_shares, hedge_parameters):
    hedgeProfits = short_sell(start, end, long_etf,
                              current_shares, hedge_parameters.hedging_percentage, hedge_parameters.interest_rate)

    price_at_hedge_end = long_etf.loc[end].values[0]
    # If profits from hedging are positive move again to the position
    if hedgeProfits > 0:
        cash += hedgeProfits
        new_shares, cash = buy(cash, price_at_hedge_end)
        current_shares += new_shares
    else:
        shares = math.ceil(abs(hedgeProfits) / price_at_hedge_end)
        current_shares -= shares
        residual = shares * price_at_hedge_end - hedgeProfits
        cash += residual
        assert(current_shares >= 0)

    return cash, current_shares


class Bond_strategy_parameters:
    def __init__(self, short_etf, ratio=0.4):
        self.ratio = ratio
        self.short_etf = short_etf


def bond_hedge_strategy(long_etf, start, end, cash, current_shares, hedge_parameters):
    """
        Strategy:
        input ratio between
        1. Liquidate ration% of your total capital
        2. Use this capital to buy shares from the sort etf
        3. Sell the short ETF
        4. Re-Invest
    """

    # Sell long ETF - Start
    price_long_etf_start = long_etf.loc[start].values[0]
    total_capital = current_shares * price_long_etf_start

    liquidation_amount = hedge_parameters.ratio * total_capital
    shares_sold = math.floor(liquidation_amount / price_long_etf_start)
    current_shares -= shares_sold
    cash = shares_sold * price_long_etf_start

    # Buy Short ETF - Start
    short_etf_price_start = hedge_parameters.short_etf.loc[start].values[0]
    short_etf_shares, cash = buy(cash, short_etf_price_start)

    # Sell short ETF - End
    short_etf_price_end = hedge_parameters.short_etf.loc[end].values[0]
    cash += short_etf_shares * short_etf_price_end

    # Buy long ETF - End
    long_etf_price_end = long_etf.loc[end].values[0]
    rebought_shares, cash = buy(cash, long_etf_price_end)
    assert(cash > 0)
    return cash, rebought_shares + current_shares


def main_loop(long_etf, strategy, initial_investment, downtrend_detector, downtrend_detector_parameters, hedge_stragegy, hedge_parameters, fund_days):

    # Calculate starting investment and baseline performance
    price_start = long_etf.iloc[0].values[0]
    current_shares, cash = buy(initial_investment, price_start)

    price_end = long_etf.iloc[-1].values[0]
    baseline_value = price_end * current_shares
    baseline_carg = cagr(baseline_value, initial_investment, fund_days)

    # Use the triggers to find all the periods that we are bearish
    bear_period_start, bear_period_end = downtrend_detector(
        long_etf, downtrend_detector_parameters)

    for i in range(0, min(bear_period_start.shape[0], bear_period_end.shape[0])):
        start = bear_period_start.iloc[i].name
        price_start = long_etf.loc[start].values[0]
        valuation_start = cash + current_shares * price_start
        end = bear_period_end.iloc[i].name
        price_end = long_etf.loc[start].values[0]

        cash, current_shares = hedge_stragegy(long_etf, start, end,
                                              cash, current_shares, hedge_parameters)

        valuation_end = cash + current_shares * price_end

        logger.debug("Bear {} {}".format(
            start.strftime('%B %Y'), end.strftime('%B %Y')))
        logger.debug(
            "\t  Valuation(start -> end) {:.2f} -> {:.2f}".format(valuation_start, valuation_end))
        logger.debug(
            "\t  Cost: {:.2f}".format(valuation_end - valuation_start))

    price_end = long_etf.iloc[-1].values[0]
    strategy_portfolio_value = price_end * current_shares
    strategy_value = strategy_portfolio_value + cash
    hedged_cagr = cagr(strategy_value, initial_investment, fund_days)

    logger.info("Results Overview")
    logger.info("\t- Fund duration {:.2f} years".format(fund_days/365))
    logger.info("\t- Long & Hold(Baseline Strategy) Valuation: {:.2f}, CAGR: {:.2f}%".format(
        baseline_value, baseline_carg))
    logger.info("\t- {}(Hedging Strategy) {:.2f} (Portofolio: {:.2f} Cash: {:.2f}) and a CAGR of {:.2f}%".format(
        strategy, strategy_value, strategy_portfolio_value, cash, hedged_cagr))

    logger.info("\t- {} outperforms Baseline by {:.2f}x".format(strategy,
                                                                strategy_value/baseline_value))
    return hedged_cagr


def read_dataset(path, columns):
    path = path.lower()
    try:
        df = pd.read_csv(
            path, usecols=columns, header=0, parse_dates=[0], index_col=0, skipinitialspace=True)
    except:
        logger.error("ERROR: Failed to load {} as CSV".format(path))
        logger.error(
            "Make sure it follows the format (Date, Close/Last, Volume, Open, High, Low)")
        logger.error(
            "You can download the data from https://www.nasdaq.com/market-activity/funds-and-etfs/iwo/historical")
        exit(1)
    return df


def correlation_under_signal(long_etf, short_etf, signal, signal_paremeters, visualize):

    bear_period_start, bear_period_end = signal(
        long_etf, signal_paremeters)

    cor = []
    for i in range(0, min(bear_period_start.shape[0], bear_period_end.shape[0])):
        start = bear_period_start.iloc[i].name
        end = bear_period_end.iloc[i].name
        curr_long = long_etf.loc[start:end]
        curr_short = short_etf.loc[start:end]
        joined = curr_long.join(curr_short,
                                lsuffix='_long', rsuffix='_short')

        corr_df = joined.corr(method='pearson')
        correlation = corr_df['Close_short'][0]
        cor.append(float(correlation))

    df = pd.DataFrame({'Correlation under Signal': cor})
    logger.info(df.describe())
    if visualize:
        df.hist()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Line Interface')
    parser.add_argument('-d', '--dataset', type=str,
                        nargs='?', help='Path to the dataset')
    parser.add_argument('-b', '--hedgeDataset', type=str,
                        nargs='?', help='Path to the short dataset')
    parser.add_argument('-v', '--visualize', action='store_true', default=False,
                        help='Visualize data')
    parser.add_argument('-p', '--optimize', action='store_true', default=False,
                        help='Optimize the hyper-parameters of the simulation')
    parser.add_argument('-s', '--hedgeStrategy', type=str,
                        nargs='?', help='Hedging Strategy')
    parser.add_argument('-l', '--signal', type=str, default="ema",
                        nargs='?', help='Bear Signal')
    args = parser.parse_args()

    col_list = ["Date", price_column]

    historicalData = read_dataset(args.dataset, col_list)
    long_etf = historicalData.sort_index()

    logger.info("Preprocessing the dataset")

    fund_days = (long_etf.iloc[-1].name - long_etf.iloc[0].name).days

    signal_paremeters = None
    signal = None
    if args.signal.lower() == "ema":
        signal = ema_crossover_signal
        signal_paremeters = Ema_crossover_parameters()
    else:
        logger.error("Signal not supported. Program will terminate")
        exit(1)

    if args.visualize:
        visualize_signal(long_etf, signal_paremeters)

    hedge_strategy, hedge_strategy_paremeters, strategy = None, None, None
    hedge_strategy_name = args.hedgeStrategy.lower()
    if hedge_strategy_name == "bond" or hedge_strategy_name == "bonds":
        hedgeDataset = read_dataset(args.hedgeDataset, col_list)
        hedgeDataset = hedgeDataset.sort_index()
        hedge_strategy = bond_hedge_strategy
        hedge_strategy_paremeters = Bond_strategy_parameters(hedgeDataset)
        correlation_under_signal(
            long_etf, hedgeDataset, signal, signal_paremeters, args.visualize)
    elif hedge_strategy_name == "short sell" or hedge_strategy_name == "short":
        hedge_strategy_paremeters = Short_sell_paremeters()
        hedge_strategy = short_sell_strategy
    else:
        logger.error("Hedge strategy not supported. Program will terminate")
        exit(1)

    # TODO: Figure out hyper parameter optimization
    main_loop(long_etf, args.hedgeStrategy, initial_investment,
              signal, signal_paremeters, hedge_strategy, hedge_strategy_paremeters, fund_days)
