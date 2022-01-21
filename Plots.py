import numpy as np
import pandas as pd
import altair as alt
from pyecharts import options as opts
from pyecharts.charts import Line


def alt_lines(df):
    source = df.round(3).reset_index().melt('index', var_name='Portfolio', value_name='y')

    # Create a selection that chooses the nearest point & selects based on x-value
    # nearest = alt.selection(type='single', nearest=True, on='mouseover',
    #                         fields=['index'], empty='none')
    # The basic line
    line = alt.Chart(source).mark_line(interpolate='basis').encode(
        x=alt.X('index:T', title=None, axis=alt.Axis(format="%Y-%m-%d")),
        y=alt.Y('y:Q', title=None),
        color=alt.Color('Portfolio:N', title=None)
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    # selectors = alt.Chart(source).mark_point().encode(
    #     x='index:T',
    #     opacity=alt.value(0),
    # ).add_selection(
    #     nearest
    # )
    # # Draw points on the line, and highlight based on selection
    # points = line.mark_point().encode(
    #     opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    # )
    # # Draw text labels near the points, and highlight based on selection
    # text = line.mark_text(align='left', dx=5, dy=-5).encode(
    #     text=alt.condition(nearest, 'y:Q', alt.value(' '))
    # )
    # # Draw a rule at the location of the selection
    # rules = alt.Chart(source).mark_rule(color='gray').encode(
    #     x='index:T',
    # ).transform_filter(
    #     nearest
    # )
    # # Put the five layers into a chart and bind the data
    # p = alt.layer(
    #     line, selectors, points, rules, text
    # ).properties(
    #     width=900
    # )
    return line.properties(width=800)


def alt_areas(df):
    source = df.round(3).reset_index().melt('index', var_name='Portfolio', value_name='y')
    area = alt.Chart(source).mark_area(interpolate='basis', opacity=0.8).encode(
        x=alt.X('index:T', title=None, axis=alt.Axis(format="%Y-%m-%d")),
        y=alt.Y('y:Q', title=None),
        color=alt.Color('Portfolio:N', title=None)
    )
    return area.properties(width=800)


def pye_lines(data) -> Line:
    if type(data) is pd.Series:
        data = pd.DataFrame(data)
    dts = data.index
    if type(dts[0]) is pd.Timestamp:
        dts = tuple(pd.Timestamp.strftime(d, format='%Y-%m-%d') for d in dts)
    keys = data.columns.tolist()
    l = Line(init_opts=opts.InitOpts(width="800px", height="800px"))
    l.add_xaxis(xaxis_data=dts)
    for k in keys:
        l.add_yaxis(
            series_name=k,
            y_axis=np.round(data.loc[:, k].values, 3),
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
        )
    l.set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        datazoom_opts=[
            opts.DataZoomOpts(),
            opts.DataZoomOpts(type_="inside")
        ],
    )
    return l

def pye_areas(data) -> Line:
    if type(data) is pd.Series:
        data = pd.DataFrame(data)
    dts = data.index
    if type(dts[0]) is pd.Timestamp:
        dts = tuple(pd.Timestamp.strftime(d, format='%Y-%m-%d') for d in dts)
    keys = data.columns.tolist()
    l = Line(init_opts=opts.InitOpts(width="800px", height="800px"))
    l.add_xaxis(xaxis_data=dts)
    for k in keys:
        l.add_yaxis(
            series_name=k,
            y_axis=np.round(data.loc[:, k].values, 3),
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
        )
    l.set_series_opts(areastyle_opts=opts.AreaStyleOpts(opacity=0.5))
    l.set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        datazoom_opts=[
            opts.DataZoomOpts(),
            opts.DataZoomOpts(type_="inside")
        ],
    )
    return l
