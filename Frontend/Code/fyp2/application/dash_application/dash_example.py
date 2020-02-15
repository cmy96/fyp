"""Create a Dash app within a Flask app."""
from pathlib import Path
import dash
import dash_table
import dash_html_components as html
import pandas as pd
from .layout import html_layout
import json
import datetime
import pandas as pd
import plotly.graph_objects as go
import dash_core_components as dcc
from dash.dependencies import Input, Output
from textwrap import dedent as d
import numpy as np
import operator

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#df = pd.read_csv("data/clinical.csv")

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True


def Add_Dash(server):
    """Create a Dash app."""
    external_stylesheets = ['/static/dist/css/styles.css',
                            'https://fonts.googleapis.com/css?family=Lato',
                            'https://use.fontawesome.com/releases/v5.8.1/css/all.css']
    external_scripts = ['/static/dist/js/includes/jquery.min.js',
                        '/static/dist/js/main.js']
    dash_app = dash.Dash(server=server,
                         external_stylesheets=external_stylesheets,
                         external_scripts=external_scripts,
                         routes_pathname_prefix='/dashapp/')

    # Override the underlying HTML template
    #dash_app.index_string = html_layout

    # Create Dash Layout comprised of Data Tables
    dash_app.layout = html.Div(

        #children=get_datasets(),
        #id='dash-container',

        children = [

            html.Div(
                html.Nav(className = "nav nav-pills", children = [
                html.A('C.A.R.E', className="nav-item nav-link btn", href='/'),
                html.A('Calculator', className="nav-item nav-link active btn", href='/dashapp') 
                ]
                )
            ),
            html.H1("lalallaala"),
            html.H1("lallalalalala"),

            html.Div(
                dcc.Graph(
                    id='example-graph',
                    figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                        {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                    ],
                    'layout': {
                    'title': 'Dash Data Visualization'
                    }
                    }
                )
            )
        ]
    )

    return dash_app.server


# def get_datasets():
#     """Return previews of all CSVs saved in /data directory."""
#     p = Path('.')
#     data_filepath = list(p.glob('data/clinical.csv'))
#     arr = ['This is an example Plot.ly Dash App.']
#     for index, csv in enumerate(data_filepath):
#         df2 = pd.read_csv(data_filepath[index]).head(2)
#         table_preview = dash_table.DataTable(
#             id='table_' + str(index),
#             columns=[{"name": i, "id": i} for i in df2.columns],
#             data=df2.to_dict("rows"),
#             sort_action="native",
#             sort_mode='single'
#         )
#         arr.append(table_preview)
#     return arr




