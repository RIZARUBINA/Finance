import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
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

def rand_weights(n):  
        ''' Produces n random weights that sum to 1 '''  
        k = np.random.rand(n)  
        return k / sum(k)

def random_portfolio(portfolio_R):  
    ''' Returns the mean and standard deviation of returns for a random portfolio '''
    r = np.mean(portfolio_R, axis=0)
    x = rand_weights(portfolio_R.shape[1])
    C = np.cov(portfolio_R.values.T)
    mu = x @ r
    sigma = np.sqrt(x @ C @ x)
    return mu, sigma  


def portfolio_std(x, C):
    '''
    Standard deviation of specified portfolio by covariance matrix C
    and (x1,..,xn)
    '''
    return np.sqrt(x @ C @ x)

    
def portfolio_var(x, C):
    ''' Volatility of specified portfolio by covariance matrix C and (x1,..,xn) '''
    return x @ C @ x

def portfolio_return(x, R):
    ''' Return of specified portfolio by mean returns and (x1,..,xn) '''
    return x @ R

def portfolio_performance(x, R, C):
    '''Return and srd of specified portfolio'''
    r = x @ R
    sigma = np.sqrt(x @ C @ x)
    return r, sigma


def make_efficient_portfolio(R_mean, C, target, by=portfolio_std, short_terms = False):
    '''
    Find efficient on specified function portfolio. 
    Returns solution of optimization task
    '''
    assets_num = len(R_mean)
    # x0 = assets_num*[1./assets_num] # init solution, diversified portfolio
    #x0 = np.array([0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])
    x0 = rand_weights(assets_num)
    args = (C)
    bound = (None, None) if short_terms else (0.0, 1.0)
    bounds = tuple(bound for asset in range(assets_num)) # 0 <= x_i <=1 restricted short-terms
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, # constraint for sum(x) = 1
                   {'type': 'eq', 'fun': lambda x: portfolio_return(x, R_mean) - target} # constraint for return
                  )

    efficient_portfolio_sol = minimize(by, x0=x0, method= 'SLSQP',
        args= args, bounds=bounds, constraints=constraints
    )
    return efficient_portfolio_sol

def efficient_frontier(R_mean, C, returns_range, by=portfolio_std, short_terms = False):
    '''Create efficient frontier'''
    efficients = [ make_efficient_portfolio(R_mean, C, ret) for ret in returns_range]
    return np.column_stack([
        [p['fun'] for p in efficients],
        returns_range
    ])


def utility_function(x, R_mean, C, gamma):
    '''Utility function to make optimal portfolio. Need to be minimized'''
    sigma = portfolio_std(x, C)
    E = portfolio_return(x, R_mean)
    return E - gamma * sigma

def rev_utility_function(x, R_mean, C, gamma):
    '''Revert Utility function to make optimal portfolio, Need to be maximized'''
    sigma = portfolio_std(x, C)
    E = portfolio_return(x, R_mean)
    return gamma * sigma - E

def make_opt_portfolio(R_mean, C, gamma,short_terms = False):
    '''
    Find optimal portfolio by minimizing utility function
    Returns solution of optimization task
    '''
    assets_num = len(R_mean)
    x0 = rand_weights(assets_num)
    args = (R_mean, C, gamma)
    bound = (None, None) if short_terms else (0.0, 1.0)
    bounds = tuple(bound for asset in range(assets_num)) # 0 <= x_i <=1 restricted short-terms
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # constraint for sum(x) = 1
    opt_portfolio_sol = minimize(rev_utility_function, x0=x0, method= 'SLSQP',
        args= args, bounds=bounds,  constraints=constraints
    )
    return opt_portfolio_sol


def optimal_portfolios(R_mean, C, gamma_range,short_terms = False):
    '''Create optimal portfolios for different gammas'''
    opt = [ make_opt_portfolio(R_mean, C, gamma,short_terms) for gamma in gamma_range]
    return np.array([ [portfolio_std(p.x, C), portfolio_return(p.x, R_mean)] for p in opt])


def rev_sharpe_ratio(x, R_mean, Ef, C):
    E,sigma = portfolio_performance(x, R_mean, C)
    return -(E - Ef) / sigma

def make_opt_portfolio_by_sharpe_ratio(R_mean, C, Ef, short_terms = False):
    '''
    Find optimal portfolio by maximizing sharpe ratio.
    Returns solution of optimization task
    '''
    assets_num = len(R_mean)
    #x0 = rand_weights(assets_num)
    x0 = assets_num*[1./assets_num] # init solution, diversified portfolio
    args = (R_mean, Ef, C)
    bound = (None, None) if short_terms else (0.0, 1.0)
    bounds = tuple(bound for asset in range(assets_num)) # 0 <= x_i <=1 restricted short-terms
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # constraint for sum(x) = 1
    opt_portfolio_sol = minimize(rev_sharpe_ratio, x0=x0, method= 'SLSQP',
        args= args, bounds=bounds,  constraints=constraints
    )
    return opt_portfolio_sol


