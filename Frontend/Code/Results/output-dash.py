import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
from matplotlib import cm
from pywaffle import Waffle
from plotly.tools import mpl_to_plotly
import requests
import io
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Launch App
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

# Dash layout with Graphs
app.layout = html.Div([
    #HEADER#
    html.Div(
        [
            html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url('logo.png'),
                            id="logo",
                            style={
                                "height": "150px",
                                "width": "auto",
                                "margin-bottom": "25px",
                                "float":"left"
                            },
                        )
                    ], className = "one-third column",
                ),
                html.Div(
                    [
                    html.H1('Predictive Analysis', style={'textAlign': 'center', 'color': '#0080FF', 'margin-top':'30px'})
                    ],
                    className="one-half column", id="title"
                )
        ]),
        html.H1("Patient View Tab Content"),
        html.Img(
            src=app.get_asset_url('waffle-1-chart.png'),
            id="waffle-1",
            style={
                "height": "200px",
                "width": "800px",
                "margin-bottom": "0px",
                
            },
        ),
        html.P('This chart depicts the survival rates of 100 women who already had surgery. This display shows the outcomes for those women based on the inputs and treatments 1/2/5/10 [Selected] years after surgery'),

        html.Img(
            src=app.get_asset_url('waffle-2-chart.png'),
            id="waffle-2",
            style={
                "height": "200px",
                "width": "800px",
            },
        ),

        html.P('In 2017, Out of every 100 women, 49 receive chemotherapy.'),

        html.Img(
            src=app.get_asset_url('waffle-3-chart.png'),
            id="waffle-3",
            style={
                "height": "200px",
                "width": "800px",
            },
        ),

        html.Img(
            src=app.get_asset_url('waffle-4-chart.png'),
            id="waffle-4",
            style={
                "height": "200px",
                "width": "800px",
            },
        ),

        html.Img(
            src=app.get_asset_url('waffle-5-chart.png'),
            id="waffle-5",
            style={
                "height": "200px",
                "width": "800px",
            },
        ),

        html.H1("Doctor View Tab Content"),

        html.Img(
            src=app.get_asset_url('kaplan-meier.png'),
            id="kpm",
            style={
                "height": "300px",
                "width": "400px",
                "bbox_inches":"tight"
                                
            },
        )
       
        
                        


],
id="mainContainer",
style={"display": "flex", "flex-direction": "column"}
)

#server
if __name__ == '__main__':
    app.run_server(debug=True, port=5001, dev_tools_hot_reload_max_retry = 10)