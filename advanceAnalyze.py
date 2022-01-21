import h5py
import numpy as np
import pandas as pd
from functools import cache
from sklearn.linear_model import LinearRegression

analye_base_pack = h5py.File("someData.h5", "r")

ind_dict = dict(zip(np.arange(6101,
                              6135),
                    ('农林牧渔',
                     '采掘',
                     '化工',
                     '钢铁',
                     '有色金属',
                     '建筑建材（旧）',
                     '机械设备（旧）',
                     '电子',
                     '交运设备（旧）',
                     '信息设备（旧）',
                     '家用电器',
                     '食品饮料',
                     '纺织服装',
                     '轻工制造',
                     '医药生物',
                     '公用事业',
                     '交通运输',
                     '房地产',
                     '金融服务（旧）',
                     '商业贸易',
                     '休闲服务',
                     '信息服务（旧）',
                     '综合',
                     '建筑材料',
                     '建筑装饰',
                     '电气设备',
                     '机械设备',
                     '国防军工',
                     '汽车',
                     '计算机',
                     '传媒',
                     '通信',
                     '银行',
                     '非银金融')))


@cache
def sw_first_ind():
    return pd.DataFrame(analye_base_pack["sw_ind"][:],
                        index=pd.to_datetime(analye_base_pack["sw_ind"].attrs['row'].astype(str)),
                        columns=np.char.decode(analye_base_pack["sw_ind"].attrs['col']))


def board_distribution(signal_df):
    stocks = signal_df.columns
    stocks.str.slice(0, 3)
    boards = pd.DataFrame(np.char.decode(analye_base_pack['board'][:]),
                          columns=np.char.decode(analye_base_pack['board'].attrs['col'])).set_index('S_INFO_WINDCODE')
    boards_map = signal_df.T.copy()
    boards_map.index = boards.loc[stocks].values.reshape(-1)
    boards_map = boards_map.groupby(
        boards_map.index).sum().reset_index().melt('index')
    return boards_map.rename(
        columns={
            'index': 'category',
            'variable': 'x',
            'value': 'y'})


def mv_style_analysis(return_series):
    dates = return_series.index
    if isinstance(dates[0], pd.Timestamp):
        first_d = pd.Timestamp.strftime(dates[0], format="%Y%m%d")
        last_d = pd.Timestamp.strftime(dates[-1], format="%Y%m%d")
    else:
        first_d = dates[0]
        last_d = dates[-1]
    index_returns = pd.DataFrame(analye_base_pack["benchmarks"][:, 2:],
                                 index=pd.to_datetime(analye_base_pack["benchmarks"].attrs['row'].astype(str)),
                                 columns=np.char.decode(analye_base_pack["benchmarks"].attrs['col'][2:]))
    index_returns = index_returns.rename(
        columns={
            'CN2372.CNI': '大盘成长',
            'CN2373.CNI': '大盘价值',
            'CN2376.CNI': '小盘成长',
            'CN2377.CNI': '小盘价值'})
    index_returns.index = pd.to_datetime(index_returns.index.astype(str))
    style_percent = pd.concat([index_returns, return_series], axis=1).dropna().resample('Q').apply(
        lambda d: ols_param(d, return_series.name)).resample("Q").sum().apply(lambda sr: sr / (sr.sum() + 1e-16),
                                                                              axis=1)
    return style_percent.T.reset_index().melt('index').rename(
        columns={'index': 'category', 'variable': 'x', 'value': 'y'})


def ind_distribution(weights_map):
    sigs_map = np.heaviside(weights_map, 0)
    sw_class = sw_first_ind()
    sw_class = pd.DataFrame(sw_class.loc[:, sigs_map.columns.intersection(sw_class.columns)],
                            index=sw_class.index.union(sigs_map.index)).sort_index().fillna(method='ffill')
    sw_class = sw_class * sigs_map
    sw_class = sw_class.reset_index().melt('index', var_name='S_INFO_WINDCODE', value_name='IND').replace(0,
                                                                                                          np.nan).dropna()
    # 信号数量
    sw_class_count = sw_class.groupby(['index', 'IND']).count().reset_index().pivot(index='index', columns='IND',
                                                                                    values='S_INFO_WINDCODE')
    sw_class_count.columns = np.vectorize(lambda c: ind_dict[int(c)])(sw_class_count.columns)
    sw_class_count = sw_class_count.resample('2Q').sum()
    sw_class_count = sw_class_count / sw_class_count.sum(axis=1).values.reshape(-1, 1)
    sw_class_count = sw_class_count.reset_index().melt('index')
    # 市值占比
    sw_class_value = sw_class.copy()
    del sw_class
    sw_class_value['VAL'] = np.apply_along_axis(lambda arr: weights_map.loc[arr[0], arr[1]], 1,
                                                sw_class_value.values[:, :-1])
    sw_class_value = sw_class_value.groupby(['index', 'IND']).sum().reset_index().pivot(index='index', columns='IND',
                                                                                        values='VAL')
    sw_class_value.columns = np.vectorize(lambda c: ind_dict[int(c)])(sw_class_value.columns)
    sw_class_value = sw_class_value.resample('2Q').sum()
    sw_class_value = sw_class_value / sw_class_value.sum(axis=1).values.reshape(-1, 1)
    sw_class_value = sw_class_value.reset_index().melt('index')
    return sw_class_count.rename(columns={'variable': 'category', 'index': 'x', 'value': 'y'}), sw_class_value.rename(
        columns={'variable': 'category', 'index': 'x', 'value': 'y'})


def alt_stack_area(df):
    import altair as alt
    alt.data_transformers.disable_max_rows()
    sa = alt.Chart(
        df,
        width=800).mark_area().encode(
        x=alt.X(
            'x:T',
            title=None,
            axis=alt.Axis(
                format="%Y-%m-%d")),
        y=alt.Y(
            'y:Q',
            title=None,
            stack="normalize"),
        color=alt.Color('category:N', scale=alt.Scale(scheme='tableau20')),
        tooltip=[
            'x:T',
            'y:Q',
            'category:N'])
    return sa.interactive(bind_y=False)


def pye_stack_area(df):
    from pyecharts import options as opts
    from pyecharts.charts import Line

    df_wide = df.pivot(index='x', columns='category', values='y')
    c = Line()
    x_axis = df_wide.index
    if isinstance(x_axis[0], pd.Timestamp):
        x_axis = tuple(pd.Timestamp.strftime(d, format='%Y-%m-%d')
                       for d in x_axis)
    category = df_wide.columns.values
    c.add_xaxis(x_axis)
    for cat in category:
        c.add_yaxis(cat, df_wide.loc[:, cat].values, stack="stack1")
    c.set_series_opts(
        areastyle_opts=opts.AreaStyleOpts(
            opacity=1), label_opts=opts.LabelOpts(
            is_show=False))
    c.set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        datazoom_opts=[
            opts.DataZoomOpts(),
            opts.DataZoomOpts(type_="inside")
        ],
    )
    return c


def ols_param(df, col_name):
    X = df[['大盘成长', '大盘价值', '小盘成长', '小盘价值']]
    if len(X) > 1:
        X = (X - X.mean()) / X.std()
    y = df[col_name]
    l = LinearRegression()
    return pd.DataFrame(
        (l.fit(
            X,
            y).coef_ *
         X).abs(),
        columns=[
            '大盘成长',
            '大盘价值',
            '小盘成长',
            '小盘价值'],
        index=df.index)
