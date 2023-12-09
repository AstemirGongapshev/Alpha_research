import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


close = pd.read_csv('.\Close.csv', index_col=0).T
opn = pd.read_csv('.\Open.csv', index_col=0).T
high = pd.read_csv('.\High.csv', index_col=0).T
low = pd.read_csv('.\Low.csv', index_col=0).T
volume = pd.read_csv('.\Volume.csv', index_col=0).T


close.index = pd.to_datetime(close.index)
opn.index = pd.to_datetime(opn.index)
high.index = pd.to_datetime(high.index)
low.index = pd.to_datetime(low.index)
volume.index = pd.to_datetime(volume.index)

# Инфраструктура для "Aльф" АТС(Алгоритмическая торговая стратегия)

def neutralize(alpha) -> pd.DataFrame:
    return (alpha.T - alpha.T.mean()).T
    

def normalize(alpha) -> pd.DataFrame:
    return (alpha.T/alpha.T.abs().sum(axis=0)).T
    


def truncate(df, threshold=0.01):
    trunc_alpha = df.copy()

    trunc_alpha[trunc_alpha > threshold] = trunc_alpha * threshold  * trunc_alpha[trunc_alpha > threshold].apply(np.sign)
    
    long_sum = trunc_alpha[trunc_alpha > 0].sum()
    short_sum = trunc_alpha[trunc_alpha < 0].sum()

    trunc_alpha[trunc_alpha > 0] /= long_sum * 0.5
    trunc_alpha[trunc_alpha < 0] /= -short_sum * 0.5

    return trunc_alpha



def calc_drawdown(get_pnl):
    max_dr, max_cpnl = 0,0
    
    for i in get_pnl:
        max_cpnl = max(i, max_cpnl)
        max_dr = max(max_cpnl - i, max_dr)

    return max_dr



def prof(alpha, data=close):
    prof = alpha.shift(1)*(data.pct_change() - 1)
    return  prof.sum(axis=1)



def prof_year(prof):
    return prof.groupby(prof.index.year).apply(lambda x: x.sum())



def ret_matrix(close):
    return (close / close.shift(axis=1) - 1).fillna(0)



def get_cumpnl(prof):
    return prof.cumsum()



def cumpnl(alpha, close_=close):
    return prof(alpha, close).cumsum()



def get_cumpnl_year(prof):
    return prof.cumsum().groupby(prof.index.year).apply(lambda x: x.sum())



def volatility(prof):
    return prof.std(axis=0).sort_values()



def calc_sharpe(prof):
    return np.sqrt(len(prof))*prof.mean()/prof.std()



def turnover(alpha):
    return (alpha.shift(1) - alpha).abs().sum()



def cut_middle(alpha, n):
    return np.where((alpha > np.quantile(alpha, 0.5 - n)) & (alpha < np.quantile(alpha, 0.5 + n)), 0, alpha)



def cut_outliers(alpha, n):
    alpha[(alpha > np.quantile(alpha, n)) | (alpha < np.quantile(alpha, 1 - n))] = 0
    return alpha



def corr(prof_alpha1,prof_alpha2):
    return prof_alpha1.corr(prof_alpha2)



def rank_vector(vector):
    N = len(vector)
    sorted_indices = sorted(range(N), key=lambda i: vector[i])
    
    rank_vector = [sorted_indices.index(i) / (N - 1) for i in range(N)]

    return rank_vector



def get_drawdown_years(pnl_cum):
    return pnl_cum.groupby(pnl_cum.index.year).apply(lambda x: calc_drawdown(x))



def get_sharpe_year(prof):
    return prof.groupby(prof.index.year).apply(lambda x: calc_sharpe(x))



def sharpe(pnl, year=0): 
    t = 252
    sharpe = ( pnl[t*(year):t*(year+1)].mean()/pnl[t*(year):t*(year+1)].std() ) * (t**0.5)
    return sharpe


def sharpe(pnl, year=0): 
    t = 252
    sharpe = ( pnl[t*(year):t*(year+1)].mean()/pnl[t*(year):t*(year+1)].std() ) * (t**0.5)
    return sharpe



def decay(alpha, n):
  return alpha.ewm(n).mean()


def AlphaStats(alpha, data=close):
    
    profit = prof(alpha, data).cumsum()
    profit_year = prof_year(prof(alpha, data))
    sharpe_coef_year = get_sharpe_year(prof(alpha, data))
    profit_year_max = profit_year.max()  
    max_drow_year = (get_drawdown_years(get_cumpnl(prof(alpha, close))).sort_values()).max()
    turnovers = turnover(alpha).mean()

    stats_df = pd.DataFrame({
        'Profit': [profit[profit.index.year == year].iloc[-1] for year in range(2010, 2015)],
        'Profit_year': profit_year,
        'Sharpe Ratio': sharpe_coef_year,
        'Max Profit Year': profit_year_max, 
        'Max Drawdown': max_drow_year,
        'Turnovers(Mean)': turnovers
    })

    stats_df = stats_df.drop_duplicates().reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Profit', color=color)
    ax1.plot(profit.index, profit.values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Additional Metrics', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    for year in pd.to_datetime(profit.index).year.unique():
        ax1.axvline(pd.Timestamp(f'{year}-01-01'), linestyle='--', color='gray', alpha=0.5)

    plt.show()
    return stats_df



