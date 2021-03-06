from plotly.graph_objs import Bar


def get_graphs(df):
    """Extracts data needed for visuals and returns figures.

    :param df: (pandas DataFrame)
    :return: (list of figures)
    """

    # Counts frequency of each genre (news, direct, social).
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)
    genre_names = [genre.capitalize() for genre in genre_names]

    # create visuals
    genres_plot = [Bar(
        x=genre_names,
        y=genre_counts,
        # marker_color="dimgrey",
    )]

    genres_layout = dict(
        {
            "title": {
                "text": "<b>Origin of Messages</b><br>Most messages arrive via the news.",
                "font": {"family": "Roboto", "size": 16},
            },
        }
    )

    categories = df.iloc[:, 4:].sum().sort_values(ascending=True)[-10:] / df.shape[0]
    label_plot = [Bar(
        x=categories.values.tolist(),
        y=categories.index.tolist(),
        orientation="h",
        # marker_color="dimgrey",
    )]

    label_layout = dict(
        {
            "title": {
                "text": "<b>10 Most Common Categories.</b>"
                        "<br>Not all messages are relevant:"
                        "<br>24% of the messages are <b>NOT</b> related to disaster events.",
                "font": {"family": "Roboto", "size": 16},
            },
            "margin": {
                "pad": 10,
                "l": 140,
                "r": 40,
                "t": 100,
                "b": 40,
            },
            "hoverlabel": {
                "font_size": 18,
                "font_family": "Roboto",
            },
            "yaxis": {"dtick": 1},
        }
    )

    figures = [dict(data=genres_plot, layout=genres_layout), dict(data=label_plot, layout=label_layout)]

    return figures

