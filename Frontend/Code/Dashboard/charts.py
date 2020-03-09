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
# graph1 = graph1.sort_values('Total Spent',ascending=False)

labels = bdf['Institution.Code'].unique()
values = bdf.groupby('Institution.Code')['Institution.Code'].count()
# print(values)
values = values.rename_axis('Hospital').reset_index(name='count')
values = values['count']

pdf = pdf.fillna(0)
before_6m = pdf['before_6m'].mean()
after_6m = pdf['after_6m'].mean()
after_1y = pdf['after_1y'].mean()
after_2y = pdf['after_2y'].mean()
after_3y = pdf['after_3y'].mean()
after_4y = pdf['after_4y'].mean()
after_5y = pdf['after_5y'].mean()
after_6y = pdf['after_6y'].mean()
after_7y = pdf['after_7y'].mean()
after_8y = pdf['after_8y'].mean()
after_9y = pdf['after_9y'].mean()
after_10y = pdf['after_10y'].mean()
price_list = [before_6m, after_6m, after_1y, after_2y,after_3y,after_4y, after_5y, after_6y, after_7y,after_8y,after_9y, after_10y]

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
        id='expenditure-scatter',
                figure={
                    'data': [
                        go.Scatter(                
                            x=graph1['Patient.ID'],
                            y=graph1['Total Spent'],
                            mode='markers',
                            marker=dict(
                                    size=12,
                                    color=np.random.randn(100000), #set color equal to a variable
                                    colorscale='Viridis', # one of plotly colorscales
                                    showscale=True
                                )
                        ),
                    ],
                    'layout': go.Layout(
                        title = "Distribution of patient expenditure",
                        xaxis = {'title': 'Patient ID'},
                        yaxis = {'title': 'Patient expenditure ($)'})
                }
                )

        ],className="three columns"),

        html.Div([
            dcc.Graph(
                id="boxplot",
                figure={
                    'data':[
                        go.Box(
                            y=graph1['Total Spent']
                        )
                    ]
                }
            )
        ]),
        html.Div([
                dcc.Graph(
                    id='hospital-pie-chart',
                    figure={
                        'data': [go.Pie(labels=labels, values=list(values))],
                        'layout': go.Layout(
                            title="Patient transactions by Institution"
                        )
                    }
                )
        ],className="four columns"),

        html.Div([
            dcc.Graph(
                id='predicted-price-chart',
                figure={
                    'data':[
                        go.Scatter(
                            x=list(pdf.columns),
                            y=price_list
                        )
                    ],
                    'layout':go.Layout(
                        title='Predicted Average Treatment Cost, Annual'
                    )
                }
            )
        ])
    ],className="seven columns")
],
id="mainContainer",
style={"display": "flex", "flex-direction": "column"})




#server
if __name__ == '__main__':
    app.run_server(debug=True, port=8060, dev_tools_hot_reload_max_retry = 10)
