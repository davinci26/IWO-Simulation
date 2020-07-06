import math


def growth(final, start):
    return (final - start)/start * 100


def cagr(end, start, days):
    return ((end / start)**(365 / days) - 1)*100


def buy(cash, price):
    assert(cash > 0)
    assert(price > 0)
    shares = math.floor(cash/price)
    return shares, cash - shares * price
