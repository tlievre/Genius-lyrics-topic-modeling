import plotly.graph_objects as go
import numpy as np

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
        df (pandas.Dataframe): pandas object.

    Returns:
        obj: plolty.figure object
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


def top_artists_decades(df, decades, rows = 2, cols = 2,
                        n_artists = 5, plotly_color = 'Plasma'):
    """multi Bar chart figure represent top artist views by decade

    Args:
        df (pandas.Dataframe): data
        decades (list): list of decades
        rows (int, optional): subplot row number. Defaults to 2.
        cols (int, optional): subplot col number. Defaults to 2.
        n_artists (int, optional): showing number of top artists. Defaults to 5.
        plotly_color (str, optional): color plotly. Defaults to 'Plasma'.

    Returns:
        obj: plotly.figure object
    """

    views_serie = df.groupby(['decade','artist'])['views'] \
            .sum().sort_values(ascending=False)

    # create en make subplot
    fig = make_subplots(rows=rows, cols=cols,
                        x_title = 'number of occurrences',
                        y_title = 'views',
                        subplot_titles = decades)

    for i in range(0,rows):
        for k in range(0,cols):
            if (i*rows + k) == len(decades):
                break
            
            # get the decade
            decade = decades[i*rows + k]

            # decade top views by artist of the selected decade
            decade_serie = views_serie.loc[decade]
            
            # get top artists by decade
            top_artists = decade_serie.index.values
        
            # get top views by decade
            top_views = decade_serie.values
            
            # add figure
            fig.add_trace(
                go.Bar(
                    y=top_artists[:n_artists][::-1],
                    x=top_views[:n_artists][::-1],
                    name=decade,
                    orientation='h', showlegend = False,
                    marker = dict(color = top_views,
                                colorscale=plotly_color)),
                i+1, k+1)
    return fig


def words_decades(df, decades):

    # create en make subplot
    fig = make_subplots(rows=1, cols=len(decades),
            x_title="tags",
            y_title="log(unique words)",
            subplot_titles = decades
    )

    for i, decade in enumerate(decades):

        # get df by decade
        df_d = df[df["decade"] == decade]

        # get unique tag of the decade
        tags = df_d['tag'].unique().tolist()
        tags.sort() # sort tags list

        for tag in tags:

            fig.add_trace(
                go.Box(
                    y=np.log(df_d[df_d['tag'] == tag]['unique_words']),
                    name=tag,
                    boxpoints='all',
                    customdata=np.stack(
                        (df_d[df_d['tag'] == tag]['title'],
                        df_d[df_d['tag'] == tag]['artist']),
                        axis=-1
                    ),
                    hovertemplate = 
                    "title: %{customdata[0]}<br>" +
                    "artist: %{customdata[1]}<br>" +
                    "log(words): %{y}",
                ), 
                1, i+1
            )
    return fig