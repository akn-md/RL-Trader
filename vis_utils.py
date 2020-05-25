import plotly.graph_objects as go


def plot_trades(data, buys, sells):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.open))

    buys_x = list(list(zip(*buys))[0])
    buys_y = list(list(zip(*buys))[1])
    sells_x = list(list(zip(*sells))[0])
    sells_y = list(list(zip(*sells))[1])
    fig.add_trace(
        go.Scatter(x=buys_x,
                   y=buys_y,
                   mode='markers',
                   name='buys',
                   fillcolor='royalblue'))
    fig.add_trace(
        go.Scatter(x=sells_x,
                   y=sells_y,
                   mode='markers',
                   name='sells',
                   fillcolor='firebrick'))

    fig.show()


def get_trade_plot(data, buys, sells):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.open))

    if len(buys) > 0:
        buys_x = list(list(zip(*buys))[0])
        buys_y = list(list(zip(*buys))[1])
        fig.add_trace(
            go.Scatter(x=buys_x,
                       y=buys_y,
                       mode='markers',
                       name='buys',
                       fillcolor='royalblue'))
    if len(sells) > 0:
        sells_x = list(list(zip(*sells))[0])
        sells_y = list(list(zip(*sells))[1])
        fig.add_trace(
            go.Scatter(x=sells_x,
                       y=sells_y,
                       mode='markers',
                       name='sells',
                       fillcolor='firebrick'))

    return fig
