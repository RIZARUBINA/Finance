import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import statistics as s


def download_by_tickers(tickers_file='tickers.txt', start="2017-01-01", end="2018-01-01"):
    with open(tickers_file, 'r') as f:
        raw = f.read()
    tickers = raw.split()

    data = yf.download(tickers, start=start, end=end)
    return data


def serialize_df(df, name, to_pickle=True, to_csv=True):
    '''Serialize df into as pickle and/or csv '''
    if to_csv:
        df.to_csv(name + ".csv")
    if to_pickle:
        df.to_pickle(name + ".pickle")


def init():
    ''' Get ready data'''
    raw_data = download_by_tickers()
    data = raw_data.dropna(axis='columns')
    dClosed = data['Close']
    dVolume = data['Volume']
    serialize_df(dVolume, 'volume_cleared')
    serialize_df(dClosed, 'close_cleared')
    serialize_df(data, 'data_cleared')


def foo():
    R = np.log(data / data.shift(1))  # find dayly returns
    R = R.drop(datetime.strptime('2017-01-02', '%Y-%m-%d'))  # drop first day
    # R['AAD.DE'] - returns vector of AAD.DE
