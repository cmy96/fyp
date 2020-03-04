### Import Libraries ###
import json 
import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output
from textwrap import dedent as d
import numpy as np
import operator
import base64

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Launch App
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True


#Create df from .csv file
bills = 'C:\\Users\\Jesslyn\\Documents\\GitHub\\fyp\\Frontend\\Code\\Dashboard\\bills.csv'
prices = 'C:\\Users\\Jesslyn\\Documents\\GitHub\\fyp\\Frontend\\Code\\Dashboard\\price_master.csv'
clinical = 'C:\\Users\\Jesslyn\\Documents\\GitHub\\fyp\\Frontend\\Code\\Dashboard\\clinical.csv'
bdf = pd.read_csv(bills, index_col=0)
pdf = pd.read_csv(prices, index_col=0)
cdf = pd.read_csv(clinical, index_col=0)


#Data Manipulation
graph1 = bdf.groupby('Patient.ID')['Gross..exclude.GST.'].sum()
graph1 = graph1.rename_axis('Patient.ID').reset_index(name='Total Spent')
graph1 = graph1.sort_values('Total Spent',ascending=False)

labels = bdf['Institution.Code'].unique()
values = bdf.groupby('Institution.Code')['Institution.Code'].count()
# print(values)
values = values.rename_axis('Hospital').reset_index(name='count')
values = values['count']


def create_cost_bins():
    bins = []
    for i in range(0,820000,100000 ):
        bins.append(i)
    return bins

cost_bins = create_cost_bins()
cost_bins = graph1.groupby(pd.cut(graph1['Total Spent'], bins=20, precision = 0, right = False)).size()

# Dash layout with Graphs
app.layout = html.Div([

#Line chart
html.Div([
    html.Div([dcc.Graph(
        id='expenditure-histogram',
                figure={
                    'data': [
                        go.Histogram(                

                            x = graph1['Total Spent'],
                            xbins=dict(start=graph1['Total Spent'].min(), end=graph1['Total Spent'].max(), size=30000),
                            text = list(cost_bins), 
                            marker = dict(color = '#97B2DE')
                        ),
                    ],
                    'layout': go.Layout(
                        title = "Distribution of patient expenditure",
                        xaxis = {'title': 'Amount spent on treatments'},
                        yaxis = {'title': 'Number of patients'},
                    )
                }
                )

        ],className="three columns"),
        html.Div([
                dcc.Graph(
                    id='hospital-pie-chart',
                    figure={
                        'data': [go.Pie(labels=labels, values=list(values))],
                        'layout': go.Layout(
                            title="Pie Chart"
                        )
                    }
                )
        ],className="four columns")

    ],className="seven columns")
],className = "container")




#server
if __name__ == '__main__':
    app.run_server(debug=True, port=8060, dev_tools_hot_reload_max_retry = 10)
