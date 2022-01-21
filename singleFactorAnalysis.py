import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_pyecharts
from analyzeStats import *
from Plots import *

import h5py

feature_test_pack = h5py.File("demo-layer10.h5", 'r')
analye_base_pack = h5py.File("someData.h5", "r")

bench = pd.DataFrame(analye_base_pack["benchmarks"][:, :2],
                     index=pd.to_datetime(analye_base_pack["benchmarks"].attrs['row'].astype(str)),
                     columns=np.char.decode(analye_base_pack["benchmarks"].attrs['col'][:2]))

# Draw sidebar
st.sidebar.title("Feature Evalutations")
pages = ('Briefing', 'Tier Top', 'Tier Bottom', 'Long-Short')

genre = st.sidebar.selectbox("", pages)

dates = feature_test_pack.attrs['row']
stocks = np.char.decode(feature_test_pack.attrs['col'])
bench.index = pd.to_datetime(bench.index)
bench = bench.loc[pd.to_datetime(np.asarray(dates).astype(str))]
# 并收益
ret_w_fee = pd.DataFrame(feature_test_pack['returns'][:, :, 0].T, index=bench.index,
                         columns=np.arange(1, 11).astype(str)[::-1])
ret_wo_fee = pd.DataFrame(feature_test_pack['returns'][:, :, 1].T, index=bench.index,
                          columns=np.arange(1, 11).astype(str)[::-1])

turns = pd.DataFrame(np.vstack((np.abs(feature_test_pack['weights'][:, 0, :]).sum(axis=1),
                                np.abs(np.diff(feature_test_pack['weights'][:], axis=1)).sum(axis=2).T)),
                     index=bench.index, columns=np.arange(1, 11).astype(str)[::-1])

holds = pd.DataFrame((feature_test_pack['positions'][:] != 0).sum(axis=2).T, index=bench.index,
                     columns=np.arange(1, 11).astype(str)[::-1])

with st.sidebar.expander("Result settings:"):
    fee_counts = st.sidebar.radio("Fee setting:", ('Count fee', 'No fee'))
cumu_type = st.sidebar.radio("Cumulative setting:", ('cumsum', 'Cumprod'))
plot_m = st.sidebar.radio("Plot module:", ("Altair", "Pyecharts"))

return_for_brief = ret_w_fee if fee_counts == 'Count fee' else ret_wo_fee
use_cumsum = True if cumu_type == 'cumsum' else False
line_renderer = lambda data: st.altair_chart(alt_lines(data)) if plot_m == 'Altair' else st_pyecharts(
    pye_lines(data))
area_renderer = lambda data: st.altair_chart(alt_areas(data)) if plot_m == 'Altair' else st_pyecharts(
    pye_areas(data))

ex_bench = st.sidebar.radio("Benchmark: ", tuple(bench.columns))


def main_page():
    st.header("Briefing")
    # 总表
    st.subheader('Absolute Return Performance')
    df = layers_table(
        return_for_brief,
        True,
        use_cumsum,
        turnover_series=turns,
        holds_series=holds)
    st.dataframe(
        df.style.format(
            formatter={
                'Ann. Return': "{:.2%}",
                'Win': "{:.2%}",
                'Max Drawdown': "{:.2%}",
                'Avg. Turn': "{:.2%}",
                'Avg. Stocks': "{:.2f}",
                'Sharpe': "{:.2f}",
                'Calmar': "{:.2f}",
                'MDD Date': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d"),
                'LDD Start': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d"),
                'LDD End': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d")}), height=1000)
    # 净值曲线
    if use_cumsum:
        line_renderer(return_for_brief.cumsum())
    else:
        line_renderer((return_for_brief + 1).cumprod())

    if True:  # 有超额
        st.subheader('Excess Returns Performance')
        df_ex = layers_table(
            return_for_brief - bench[ex_bench].values.reshape(-1, 1),
            False,
            use_cumsum)
        st.dataframe(df_ex.style.format(
            formatter={
                'Ann. Ex. Return': "{:.2%}",
                'Ex. Win': "{:.2%}",
                'Ex. Max Drawdown': "{:.2%}",
                'Info. Ratio': "{:.2f}",
                'ExRet/ExMdd': "{:.2f}",
                'Ex. MDD Date': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d"),
                'Ex. LDD Start': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d"),
                'Ex. LDD End': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d")}))
        # 净值曲线
        if use_cumsum:
            line_renderer((return_for_brief - bench[ex_bench].values.reshape(-1, 1)).cumsum())
        else:
            line_renderer((return_for_brief - bench[ex_bench].values.reshape(-1, 1) + 1).cumprod())

    st.subheader("Signal Distributions")
    """
    #### Signal Overall Coverage
    """
    all_sigs = pd.DataFrame(np.heaviside(np.abs(feature_test_pack['positions'][:]).sum(axis=0), 0), index=bench.index,
                            columns=stocks)
    num_holds = all_sigs.sum(axis=1).rename("NUM. of Stocks")
    area_renderer(num_holds)

    """
    **Sql connection required for this session.**
    """


def tier_detail(tier_name, col_name):
    st.header("Tier %s" % tier_name)

    st.subheader("Abstract Return Performance")
    df = annual_table(return_for_brief[col_name], True, use_cumsum)
    st.dataframe(
        df.style.format(
            formatter={
                'Ann. Return': "{:.2%}",
                'Win': "{:.2%}",
                'Max Drawdown': "{:.2%}",
                'Avg. Turn': "{:.2%}",
                'Sharpe': "{:.2f}",
                'Calmar': "{:.2f}",
                'MDD Date': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d"),
                'LDD Start': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d"),
                'LDD End': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d")}), height=1000)
    with st.expander("Return plots"):
        # 净值曲线
        st.subheader("Cumulative Return")
        if use_cumsum:
            line_renderer(return_for_brief[col_name].cumsum())
        else:
            line_renderer((return_for_brief[col_name] + 1).cumprod())
        # 回撤曲线
        st.subheader("Drawdown")
        if use_cumsum:
            drawdown = return_for_brief[col_name].cumsum() - return_for_brief[col_name].cumsum().expanding().max()
        else:
            drawdown = (return_for_brief[col_name] + 1).cumprod() / (
                    return_for_brief[col_name] + 1).cumprod().expanding().max() - 1
        area_renderer(drawdown)

    if True:
        ex_return = return_for_brief.loc[:, col_name] - bench[ex_bench].values
        st.subheader("Excess Returns Performance")
        df_ex = annual_table(ex_return, False, use_cumsum)
        st.dataframe(
            df_ex.style.format(
                formatter={
                    'Ann. Ex. Return': "{:.2%}",
                    'Ex. Win': "{:.2%}",
                    'Ex. Max Drawdown': "{:.2%}",
                    'Info. Ratio': "{:.2f}",
                    'ExRet/ExMdd': "{:.2f}",
                    'Ex. MDD Date': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d"),
                    'Ex. LDD Start': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d"),
                    'Ex. LDD End': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d")}), height=1000)
        with st.expander("Return plots"):
            # 净值曲线
            st.subheader("Cumulative Return")
            if use_cumsum:
                line_renderer(ex_return.cumsum())
            else:
                line_renderer((ex_return + 1).cumprod())
            # 回撤曲线
            st.subheader("Drawdown")
            if use_cumsum:
                drawdown = ex_return.cumsum() - ex_return.cumsum().expanding().max()
            else:
                drawdown = (ex_return + 1).cumprod() / (ex_return + 1).cumprod().expanding().max() - 1
            area_renderer(drawdown)

    with st.expander("Advance Analyze. Sql connection required for this session."):
        if st.button("RUN"):
            import advanceAnalyze as aa
            """
            ** Board Distribution **
            """
            signal_df = pd.DataFrame(np.heaviside(np.abs(feature_test_pack['positions'][int(col_name) - 1]), 0),
                                     index=pd.to_datetime(dates.astype(str)), columns=stocks)
            boards_df = aa.board_distribution(signal_df)
            stack_area_renderer = lambda data: st.altair_chart(
                aa.alt_stack_area(data)) if plot_m == 'Altair' else st_pyecharts(
                aa.pye_stack_area(data))
            stack_area_renderer(boards_df)

            """
            ** Market Value Style Distribution **   
            *By regressing the portfolio returns with pre-defined style benchmark returns *
            """
            style_df = aa.mv_style_analysis(ret_wo_fee[col_name]).round(2)
            stack_area_renderer = lambda data: st.altair_chart(
                aa.alt_stack_area(data)) if plot_m == 'Altair' else st_pyecharts(
                aa.pye_stack_area(data), key=str(pd.Timestamp.now()))
            stack_area_renderer(style_df)

            # industry
            weight_df = pd.DataFrame(np.heaviside(np.abs(feature_test_pack['weights'][int(col_name) - 1]), 0),
                                     index=pd.to_datetime(dates.astype(str)), columns=stocks)
            sw_inds_sig, sw_inds_val = aa.ind_distribution(weight_df)
            """
            ** Industry Signal Distribution **
            """
            stack_area_renderer(sw_inds_sig.round(3))

            """
            ** Industry Market Value Distribution **
            """
            stack_area_renderer(sw_inds_val.round(3))


def long_short_page(long_col_name, short_col_name):
    st.header("Long-Short Portfolio")
    ls_return = pd.DataFrame(
        (return_for_brief.loc[:, long_col_name] - return_for_brief.loc[:, short_col_name]).rename("LS"))
    df = annual_table(ls_return, True, use_cumsum)
    st.dataframe(
        df.style.format(
            formatter={
                'Ann. Return': "{:.2%}",
                'Win': "{:.2%}",
                'Max Drawdown': "{:.2%}",
                'Avg. Turn': "{:.2%}",
                'Sharpe': "{:.2f}",
                'Calmar': "{:.2f}",
                'MDD Date': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d"),
                'LDD Start': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d"),
                'LDD End': lambda ts: pd.Timestamp.strftime(ts, format="%Y-%m-%d")}), height=1000)
    with st.expander("Return plots"):
        # 净值曲线
        st.subheader("Cumulative Return")
        if use_cumsum:
            line_renderer(ls_return.cumsum())
        else:
            line_renderer((ls_return + 1).cumprod())
        # 回撤曲线
        st.subheader("Drawdown")
        if use_cumsum:
            drawdown = ls_return.cumsum() - ls_return.cumsum().expanding().max()
        else:
            drawdown = (ls_return + 1).cumprod() / (ls_return + 1).cumprod().expanding().max() - 1
        area_renderer(drawdown)


if genre == 'Tier Top':
    tier_detail("TOP", '1')
elif genre == 'Tier Bottom':
    tier_detail("Bottom", '10')
elif genre == 'Long-Short':
    long_short_page('1', '10')
else:
    main_page()
