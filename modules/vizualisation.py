import plotly.graph_objects as go
from plotly.subplots import make_subplots

# barplot by decade
def barplot_by_decade(df):
    """display decade frequency

    Args:
        df (obj): pandas object

    Returns:
        obj: plotly object
    """

    # groupby decade
    df_d = df.groupby(['decade']).size().reset_index(name='count')

    # create the figure
    fig = go.Figure()

    fig.add_bar(
        x=df_d.decade,
        y=df_d['count'],
        showlegend=False)

    fig.add_scatter(
            x=df_d.decade,
            y=df_d["count"],
            mode="markers+lines",
            name="trend",
            showlegend=False)

    fig.update_layout(
            title = "Music release over years",
            xaxis_title="decade",
            yaxis_title="release")
    return fig



def piechart_tags_decade(df):
    """display piechart tags

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """

    # compute tag frequencies by decade
    df_pies_d = df.groupby(['decade','tag']).size().reset_index(name='count')
    df_pies_d[df_pies_d.decade == 1960]

    # create en make subplot
    fig = make_subplots(rows=3, cols=3,
                        specs=[
                            [{'type':'domain'}
                            for i in range(1,4)] for i in range(1,4)
                        ])
    decades = df_pies_d.decade.unique().tolist()
    for i in range(0,3):
        for k in range(0,3):
            decade = decades[i*3 + k]
            # group by decade
            df_p = df_pies_d[df_pies_d.decade == decade]
            # add figure
            fig.add_trace(go.Pie(labels=df_p.tag, values=df_p['count'], name=decade), i+1, k+1)
            # add annotation
            fig.add_annotation(arg=dict(
                text=decade, x=k*0.375 + 0.125,
                y=-i*0.3927 + 0.90, font_size=10,
                showarrow=False))
            if (i*3 + k) == 6:
                break


    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")

    fig.update_layout(
        title_text="Tags proportions by decades"
        # Add annotations in the center of the donut pies.
        #annotations=[dict(text=decade, x=k*0.375+0.125, y= -i*0.125+0.90, font_size=10, showarrow=False)
        #           for k, decade in enumerate(decades) for i in range(0,4)]
    )
    return fig