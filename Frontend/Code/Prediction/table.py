import json
import datetime
import pandas as pd
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash 
from dash.dependencies import Input, Output
from textwrap import dedent as d
import numpy as np
import operator
import dash_table

# external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css']

c_output = {0.5:89, 1:172, 2:218, 5:304, 10:748}
s_output = {0.5:0.96, 1:0.92, 2:0.87, 5:0.83, 10:0.78}
df = pd.DataFrame(c_output.items(), columns=['Years', 'Cost'])
df2 = pd.DataFrame(s_output.items(), columns=['Years', 'SurvivabilityRate'])

app = dash.Dash(__name__)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll',
        'width' : '70%'
    }
}

app.layout = html.Div([
    dcc.Graph(
        id='Cost Table',
        figure={
            'data': [
                 go.Table(
                    header=dict(values=["<b>Years</b>","<b>Cost</b>"],
                    fill_color='paleturquoise',
                    align='center'),
                    cells=dict(values=[df.Years, df.Cost],
                    fill_color='white',
                    align='center')
                 )]
        }), 
    dcc.Graph(
        id='Survivability Rate Table',
        figure={
            'data': [
                 go.Table(
                    header=dict(values=["<b>Years</b>","<b>Survivability Rate</b>"],
                    fill_color='paleturquoise',
                    align='center'),
                    cells=dict(values=[df2.Years, df2.SurvivabilityRate],
                    fill_color='white',
                    align='center')
                 )],
                 })
])
if __name__ == '__main__':
    app.run_server(debug=True)