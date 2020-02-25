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

#functions to generate dict and get pecentage
# def rename_keys(dict_, new_keys):
#     """
#      new_keys: type List(), must match length of dict_
#     """
#     d1 = dict( zip( list(dict_.keys()), new_keys) )
#     return {d1[oldK]: value for oldK, value in dict_.items()}

# def calPercent(df1,columnDF,nullExist=False, *replaceWith):
#     dict_list = {}
#     values = columnDF.unique()
#     if nullExist:
#         values = np.insert(values,0,replaceWith)
#         values = np.delete(values,np.argwhere(pd.isna(values))[0][0])
#     for v in values:
#         if nullExist:
#             dict_list[v] = round((len(columnDF) - columnDF.count())/len(columnDF)*100,2)
#             nullExist = False
#             continue
#         dict_list[v] = round(len(df1[columnDF==v])/len(columnDF)*100,2)
    
    #sort values descendingly
    # dict_list = dict(sorted(dict_list.items(),key=operator.itemgetter(1),reverse = True))

    # return dict_list

#python codes for survival
s_output = {"6 months before":100.0, "6 months after":96.20, "1 year after":90.10, 
    "2 years after":86.90, "5 years after":80.09, "10 years after":71.22}
df2 = pd.DataFrame(s_output.items(), columns=['Years', 'Survival'])
#python codes for cost
c_output = {"6 months before":3882.80, "6 months after":13112.54, "1 year after":2230.19, 
    "2 years after":1736.58, "5 years after":11800.33, "10 years after":14917.57}
df = pd.DataFrame(c_output.items(), columns=['Years', 'Cost'])

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

trace1 = go.Bar(
    x=list(c_output.keys()),
    y=list(c_output.values()),
    text=list(c_output.values()),
    textposition='auto',
    name = "predicted cost"
)

trace2= go.Scatter(
    x=list(c_output.keys()),
    y=list(c_output.values()),
    name = "prediction line"
    )

trace1s = go.Bar(
    x=list(s_output.keys()),
    y=list(s_output.values()),
    text=list(s_output.values()),
    textposition='auto',
    name = "survival rate"
)

trace2s= go.Scatter(
    x=list(s_output.keys()),
    y=list(s_output.values()),
    name = "survival trendline"
    )

dashboard = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        #cost table
                     dcc.Graph(
                        id='Cost Table',
                        figure={
                        'data': [
                    go.Table(
                        header=dict(values=["<b>Years</b>","<b>Cost($)</b>"],
                        fill_color='paleturquoise',
                        align='center'),
                        cells=dict(values=[df.Years, df.Cost],
                        fill_color='white',
                        align='center')
                 )]
        })   
                     ],
                ),
                dbc.Col(
                    [
                    html.H2("Graph"), 
                    #survival table
                    dcc.Graph(
                        id='Survivability Rate Table',
                        figure={
                        'data': [
                        go.Table(
                            header=dict(values=["<b>Years</b>","<b>Survivability Rate</b>"],
                            fill_color='paleturquoise',
                            align='center'),
                            cells=dict(values=[df2.Years, df2.Survival],
                            fill_color='white',
                            align='center')
                        )]
                }),
                    ]
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                [ 
                    #survival Pie chart
                    dcc.Graph(
                        id="suvivability",
                        figure=go.Figure(
                            data=[
                                go.Pie(
                                    labels=df.Years, values=df2.Survival, hole=.3
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
                    #cost bar chart
                    dcc.Graph(
                        id="Cost Prediction",
                        figure=go.Figure(
                            data=[trace1,trace2],
                    
                            layout=go.Layout(
                                title="Patient's Cost Prediction ($)"
                            )
                        ),
                    )
                ]
            )
        ),
        dbc.Row(
            dbc.Col(
                [
                    #survival bar chart
                    dcc.Graph(
                        id="Survival Prediction",
                        figure=go.Figure(
                            data=[trace1s,trace2s],
                        layout=go.Layout(
                            title="Patient's Survival Rates Prediction (%)"
                        ),
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




