"""Create a Dash app within a Flask app."""
from pathlib import Path
import dash
import dash_table
import dash_html_components as html
import dash_bootstrap_components as dbc
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
from plotly.subplots import make_subplots
import plotly
import base64
from flask import Blueprint, render_template, request, jsonify, json, redirect, session, url_for
import flask

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#df = pd.read_csv("data/clinical.csv")


def rename_keys(dict_, new_keys):
    """
     new_keys: type List(), must match length of dict_
    """
    d1 = dict( zip( list(dict_.keys()), new_keys) )
    return {d1[oldK]: value for oldK, value in dict_.items()}

def calPercent(df1,columnDF,nullExist=False, *replaceWith):
    dict_list = {}
    values = columnDF.unique()
    if nullExist:
        values = np.insert(values,0,replaceWith)
        values = np.delete(values,np.argwhere(pd.isna(values))[0][0])
    for v in values:
        if nullExist:
            dict_list[v] = round((len(columnDF) - columnDF.count())/len(columnDF)*100,2)
            nullExist = False
            continue
        dict_list[v] = round(len(df1[columnDF==v])/len(columnDF)*100,2)
    
    #sort values descendingly
    dict_list = dict(sorted(dict_list.items(),key=operator.itemgetter(1),reverse = True))

    return dict_list


#Launch App
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True


layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview"
)
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Link", href="#")),
        dbc.DropdownMenu(
            nav=True,
            in_navbar=True,
            label="Menu",
            children=[
                dbc.DropdownMenuItem("Entry 1"),
                dbc.DropdownMenuItem("Entry 2"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Entry 3"),
            ],
        ),
    ],
    brand="Demo",
    brand_href="#",
    sticky="top",
)

labels = ["Survived", "Dead"]
values = [0.7, 0.3]

trace1 = go.Bar(
    x=["-6 Months", "Year 0", "Year 1", "Year 2", " Year 5", "Year 10"],
    y=[100,200,300,400,500,600],
    name='Surgery'
)

trace2= go.Bar(
    x=["-6 Months", "Year 0", "Year 1", "Year 2", " Year 5", "Year 10"],
    y=[50, 60, 70, 80, 90, 100],
    name='Medicine'
)


body = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Heading"),
                        html.P(
                            """\
Donec id elit non mi porta gravida at eget metus.
Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum
nibh, ut fermentum massa justo sit amet risus. Etiam porta sem
malesuada magna mollis euismod. Donec sed odio dui. Donec id elit non
mi porta gravida at eget metus. Fusce dapibus, tellus ac cursus
commodo, tortor mauris condimentum nibh, ut fermentum massa justo sit
amet risus. Etiam porta sem malesuada magna mollis euismod. Donec sed
odio dui."""
                        ),
                        dbc.Button("View details", color="secondary"),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.H2("Graph"),
                        dcc.Graph(
                            figure={"data": [{"x": [1, 2, 3], "y": [1, 4, 9]}]}
                        ),
                    ]
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                [
                    dcc.Graph(
                        id="suvivability",
                        figure=go.Figure(
                            data=[
                                go.Pie(
                                    labels=labels, values=values, hole=.3
                                    )
                                ],
                            layout=go.Layout(
                            title="Pie Chart"
                            )
                        )
                    )
                ],
            ),

        ),
        dbc.Row(
            dbc.Col(
                [
                    dcc.Graph(
                        id="Cost Prediction",
                        figure=go.Figure(
                            data=[trace1, trace2],
                            
                            layout=go.Layout(
                                barmode='stack',
                                title='Stacked Bar Chart'
                            )
                        ),
                    )
                ]
            )
        )
    ],
    className="mt-4",
)


def Add_Dash(server):
    """Create a Dash app."""
    external_stylesheets = ['.\\fyp\\Frontend\\Code\\fyp2\\application\\static\\dist\\css\\styles.css',
                            'https://fonts.googleapis.com/css?family=Lato',
                            'https://use.fontawesome.com/releases/v5.8.1/css/all.css', 
                            dbc.themes.BOOTSTRAP]
    external_scripts = ['/static/dist/js/includes/jquery.min.js',
                        '/static/dist/js/main.js']
    dash_app = dash.Dash(server=server,
                         external_stylesheets=external_stylesheets,
                         external_scripts=external_scripts,
                         #routes_pathname_prefix='/dashapp/'
                         )

    # Override the underlying HTML template
    dash_app.index_string = html_layout

    #Create Dash Layout comprised of Data Tables
    dash_app.layout = html.Div([

        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content'),
        html.Div(id='output-graph'),
        html.Div(id='input'),
        html.Div(id='output'),
    

    ])

    init_callbacks(dash_app)

    return dash_app.server



def init_callbacks(dash_app):
    @dash_app.callback(dash.dependencies.Output('page-content', 'children'),
                        
                        [dash.dependencies.Input('url', 'pathname')]
                       
                    )
    def display_page(pathname):        
        if pathname =="/dashapp/":
            return dashboard
        elif pathname == "/results/":
            cookies = session['received']
            cookies = str(cookies, 'utf-8')
            cookies =cookies.split(",")

            #print(cookies)
            cookie = html.H1(cookies)
            return results_layout
            #return dashboard


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




