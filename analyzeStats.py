import numpy as np
import pandas as pd
import streamlit as st


@st.cache
def layers_table(returns_series, is_abs=True, use_cumsum=True, *, rf=0.03, turnover_series=None, holds_series=None):
    """
    :param returns_series: pd.Dataframe, columns are layers, rows are datetime
    :param is_abs: True represents absolute revenue, else represents excess returns
    :param rf: risk-free ratio, default is 0.03
    :param use_cumsum: True for using 'cumsum' to compute cumulative returns, else using 'cumprod'
    :param turnover_series: pd.Dataframe, turnovers, same format as returns_series
    :param holds_series: pd.Dataframe, num of different stocks holds, same format as returns_series
    :return: dataframe, all info gathered
    """
    turn = None
    holds = None
    if is_abs:
        ann_ret = annualized_return(returns_series).rename('Ann. Return')
        win_ratio = win(returns_series).rename('Win')
        sharpe_ratio = sharpe(returns_series, rf).rename('Sharpe')
        calmar_ratio = calmar(returns_series, rf, cumsum=use_cumsum).rename('Calmar')
        mdd_r = annualized_mdd(returns_series, False, cumsum=use_cumsum).rename('Max Drawdown')
        mdd_d = annualized_mdd(returns_series, True, cumsum=use_cumsum).rename('MDD Date')
        ldds = ldd(returns_series, cumsum=use_cumsum).T
        if turnover_series is not None:
            turn = average_turn(turnover_series).rename('Avg. Turn')
        if holds_series is not None:
            holds = average_holds(holds_series).rename('Avg. Stocks')
    else:
        ann_ret = annualized_return(returns_series).rename('Ann. Ex. Return')
        win_ratio = win(returns_series).rename('Ex. Win')
        sharpe_ratio = sharpe(returns_series, 0).rename('Info. Ratio')
        calmar_ratio = calmar(returns_series, 0, cumsum=use_cumsum).rename('ExRet/ExMdd')
        mdd_r = annualized_mdd(returns_series, False, cumsum=use_cumsum).rename('Ex. Max Drawdown')
        mdd_d = annualized_mdd(returns_series, True, cumsum=use_cumsum).rename('Ex. MDD Date')
        ldds = ldd(returns_series, cumsum=use_cumsum).T
        ldds.columns = 'Ex. ' + ldds.columns
    tb = pd.concat([ann_ret, win_ratio, sharpe_ratio, calmar_ratio, mdd_r, mdd_d, ldds], axis=1)
    if turn is not None:
        tb = pd.concat([tb, turn], axis=1)
    if holds is not None:
        tb = pd.concat([tb, holds], axis=1)
    tb.sort_index(inplace=True)
    return tb


@st.cache
def annual_table(return_series, is_abs=True, use_cumsum=True, rf=0.03):
    """
    :param return_series: pd.Series | pd.Dataframe, columns are layers, rows are datetime
    :param is_abs: True represents absolute revenue, else represents excess returns
    :param rf: risk-free ratio, default is 0.03
    :param use_cumsum: True for using 'cumsum' to compute cumulative returns, else using 'cumprod'
    :param turnover_series: pd.Dataframe, turnovers, same format as returns_series
    :param holds_series: pd.Dataframe, num of different stocks holds, same format as returns_series
    :return: tb, dataframe, all indicators gathered
    """
    if type(return_series) is pd.Series:
        series_name = return_series.name
        return_series = pd.DataFrame(return_series, columns=[series_name])
    else:
        series_name = return_series.columns[0]
    return_series.index = pd.to_datetime(return_series.index)
    if is_abs:
        ret_per_yr = (return_series.resample("Y").mean() * 250).rename(columns={series_name: 'Ann. Return'})
        win_per_yr = return_series.resample("Y").apply(lambda s: win(s)).rename(columns={series_name: 'Win'})
        sharpe_ratio = return_series.resample("Y").apply(lambda s: sharpe(s, rf)).rename(
            columns={series_name: 'Sharpe'})
        calmar_ratio = return_series.resample("Y").apply(lambda s: calmar(s, rf, cumsum=use_cumsum)).rename(
            columns={series_name: 'Calmar'})
        mdd = return_series.resample('Y').apply(lambda s: annualized_mdd(s, cumsum=use_cumsum)).rename(
            columns={series_name: 'Max Drawdown'})
        mdd_point = return_series.resample('Y').apply(
            lambda s: annualized_mdd(s, True, cumsum=use_cumsum)).rename(
            columns={series_name: 'MDD Date'})
        ldds = \
            return_series.resample("Y").apply(lambda s: ldd(s, cumsum=use_cumsum)).reset_index().pivot(index='level_0',
                                                                                                       columns='level_1',
                                                                                                       values=series_name)[
                ['Longest Drawdown', 'LDD Start', 'LDD End']]
    else:
        ret_per_yr = (return_series.resample("Y").mean() * 250).rename(columns={series_name: 'Ann. Ex. Return'})
        win_per_yr = return_series.resample("Y").apply(lambda s: win(s)).rename(columns={series_name: 'Ex. Win'})
        sharpe_ratio = return_series.resample("Y").apply(lambda s: sharpe(s, rf)).rename(
            columns={series_name: 'Info. Ratio'})
        calmar_ratio = return_series.resample("Y").apply(lambda s: calmar(s, rf, cumsum=use_cumsum)).rename(
            columns={series_name: 'ExRet/ExMdd'})
        mdd = return_series.resample('Y').apply(lambda s: annualized_mdd(s, cumsum=use_cumsum)).rename(
            columns={series_name: 'Ex. Max Drawdown'})
        mdd_point = return_series.resample('Y').apply(
            lambda s: annualized_mdd(s, True, cumsum=use_cumsum)).rename(
            columns={series_name: 'Ex. MDD Date'})
        ldds = \
            return_series.resample("Y").apply(lambda s: ldd(s, cumsum=use_cumsum)).reset_index().pivot(index='level_0',
                                                                                                       columns='level_1',
                                                                                                       values=series_name)[
                ['Longest Drawdown', 'LDD Start', 'LDD End']]
        ldds.columns = 'Ex. ' + ldds.columns
    tb = pd.concat([ret_per_yr, win_per_yr, sharpe_ratio, calmar_ratio, mdd, mdd_point, ldds], axis=1)
    tb.index = tb.index.year.astype(str)
    total = layers_table(return_series, is_abs, use_cumsum, rf=rf)
    total.index = ['TOTAL']
    tb = pd.concat([tb, total])
    if len(return_series) > 750:
        last_3yr = layers_table(return_series.tail(750), is_abs, use_cumsum, rf=rf)
        last_3yr.index = ['LAST 750 DAYS']
        tb = pd.concat([tb, last_3yr])
    return tb


def turnover_d(weights_map):
    return pd.Series(
        [np.abs(weights_map.values[0]).sum()] + np.abs(np.diff(weights_map.values, axis=0)).sum(axis=1).tolist(),
        index=weights_map.index.values)


def stock_holds_d(position_map):
    return pd.Series((position_map.values != 0).sum(axis=1), index=position_map.index.values)


def average_turn(turnover_series):
    return turnover_series.mean()


def average_holds(holds_series):
    return holds_series.mean()


def annualized_return(return_series):
    return return_series.mean() * 250


def annualized_mdd(return_series, find_minimum_point=False, *, cumsum):
    if cumsum:
        cumulative_returns = return_series.cumsum()
        if find_minimum_point:
            return (cumulative_returns - cumulative_returns.expanding().max()).apply(
                lambda sr: sr.index.values[sr.argmin()])
        else:
            return (cumulative_returns - cumulative_returns.expanding().max()).min()
    else:
        cumulative_returns = (1 + return_series).cumprod()
        if find_minimum_point:
            return (cumulative_returns / cumulative_returns.expanding().max() - 1).apply(
                lambda sr: sr.index.values[sr.argmin()])
        else:
            return (cumulative_returns / cumulative_returns.expanding().max() - 1).min()


def annualized_volatility(return_series):
    return return_series.std() * np.sqrt(250)


def sharpe(return_series, rf):
    ann_ret = annualized_return(return_series)
    ann_vol = annualized_volatility(return_series)
    return (ann_ret - rf) / ann_vol


def calmar(return_series, rf, *, cumsum):
    ann_ret = annualized_return(return_series) - rf
    ann_mdd = -annualized_mdd(return_series, cumsum=cumsum)
    return ann_ret / ann_mdd


def win(return_series):
    return (return_series > 0).sum() / len(return_series)


def ldd(return_series, *, cumsum):
    if cumsum:
        ldds = return_series.cumsum().apply(find_ldd_sr)
    else:
        ldds = (1 + return_series).cumprod().apply(find_ldd_sr)
    ldds.index = ['Longest Drawdown', 'LDD Start', 'LDD End']
    return ldds


def find_ldd_sr(cumu_sr):
    new_highs = np.where(cumu_sr - cumu_sr.expanding().max() == 0)[0]
    if len(new_highs) <= 1:
        return 0, cumu_sr.index.values[0], cumu_sr.index.values[0]
    # 防止最后一天没有回上来
    new_highs = np.append(new_highs, [len(cumu_sr) - 1])
    ldd = np.diff(new_highs).max()
    ldd_start = np.diff(new_highs).argmax()
    ldd_end = ldd_start + 1
    return ldd, cumu_sr.index.values[new_highs[ldd_start]], cumu_sr.index.values[new_highs[ldd_end]]
