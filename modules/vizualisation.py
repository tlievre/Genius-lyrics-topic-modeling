import plotly.graph_objects as go
import numpy as np

from plotly.subplots import make_subplots

# barplot by decade
def barplot_by_decades(df, group_by = 'decade'):
    """display decade frequency by decade or double decade

    Args:
        df (obj): pandas object
        group_by (str): decade or ddecade are accepted values.

    Returns:
        obj: plotly object
    """
    
    if (group_by != 'decade') and (group_by != 'ddecade'):
        raise Exception('{} isn\'t recognize as an existing column name'.format(group_by))

    # groupby decade
    df_d = df.groupby([group_by]).size().reset_index(name='count')

    # create the figure
    fig = go.Figure()

    fig.add_bar(
        x=df_d[group_by],
        y=df_d['count'],
        showlegend=False)

    fig.add_scatter(
            x=df_d[group_by],
            y=df_d["count"],
            mode="markers+lines",
            name="trend",
            showlegend=False)

    fig.update_layout(
            title = "Music release over years",
            xaxis_title=group_by
                if decade == group_by else 'double decade',
            yaxis_title="release")
    return fig



def piechart_tags_decades(df, group_by = 'decade'):
    """display piechart tags by decade or double decade

    Args:
        df (pandas.Dataframe): pandas object.
        group_by (str): decade or ddecade are accepted values.

    Returns:
        obj: plolty.figure object
    """
    if (group_by != 'decade') and (group_by != 'ddecade'):
        raise Exception('{} isn\'t recognize as an existing column name'.format(group_by))

    # compute tag frequencies by decade
    df_pies_d = df.groupby([group_by,'tag']).size().reset_index(name='count')
    #df_pies_d[df_pies_d.decade == 1960]

    # create en make subplot
    fig = make_subplots(rows=3, cols=3,
                        specs=[
                            [{'type':'domain'}
                            for i in range(1,4)] for i in range(1,4)
                        ])
    decades = df_pies_d[group_by].unique().tolist()
    for i in range(0,3):
        for k in range(0,3):
            decade = decades[i*3 + k]
            # group by decade
            df_p = df_pies_d[df_pies_d[group_by] == decade]
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


def top_artists_decades(df, decades, rows = 2, cols = 2, group_by = 'decade',
                        n_artists = 5, plotly_color = 'Plasma',):
    """multi Bar chart figure represent top artist views by decade

    Args:
        df (pandas.Dataframe): data
        decades (list): list of decades or double decade
        rows (int, optional): subplot row number. Defaults to 2.
        cols (int, optional): subplot col number. Defaults to 2.
        n_artists (int, optional): showing number of top artists. Defaults to 5.
        plotly_color (str, optional): color plotly. Defaults to 'Plasma'.
        group_by (str): decade or ddecade are accepted values.

    Returns:
        obj: plotly.figure object
    """
    
    if (group_by != 'decade') and (group_by != 'ddecade'):
        raise Exception('{} isn\'t recognize as an existing column name'.format(group_by))

    views_serie = df.groupby([group_by,'artist'])['views'] \
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


def words_decades(df, decades, group_by = 'decade'):
    """display decade frequency by decade or double decade

    Args:
        df (obj): pandas object
        group_by (str): decade or ddecade are accepted values.

    Returns:
        obj: plotly object
    """
    
    if (group_by != 'decade') and (group_by != 'ddecade'):
        raise Exception('{} isn\'t recognize as an existing column name'.format(group_by))

    # create en make subplot
    fig = make_subplots(rows=1, cols=len(decades),
            x_title="tags",
            y_title="log(unique words)",
            subplot_titles = decades
    )

    for i, decade in enumerate(decades):

        # get df by decade
        df_d = df[df[group_by] == decade]

        # get unique tag of the decade
        tags = df_d['tag'].unique().tolist()
        tags.sort() # sort tags list

        for tag in tags:

            fig.add_trace(
                go.Box(
                    y=np.log(df_d[df_d['tag'] == tag]['unique_words']),
                    name=tag,
                    boxpoints='all',
                    marker=dict(color=[color i in range(0, len(tags)])
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
