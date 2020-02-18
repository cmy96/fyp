import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

#Launch App
app = dash.Dash(__name__)

#Create df from .csv file
url = 'data/bills1.csv'
url2 = 'data/OldFaithful.csv'
df_bills = pd.read_csv(url, index_col=0)
df = pd.read_csv(url2, index_col=0)

#Data manip for graph dataset
graph2_data = df_bills['Service.Summary..Description'].value_counts(ascending = False).head(10)
graph2_data = graph2_data.rename_axis('unique_values').reset_index(name='counts')

graph3_data = df_bills.groupby('Service.Department.Description')['Gross..exclude.GST.'].sum()
graph3_data = graph3_data.rename_axis('Service Department').reset_index(name='Treatment Gross')
graph3_data = graph3_data.sort_values('Treatment Gross',ascending=False).head(10)

# Dash layout with Graphs
app.layout = html.Div([
    html.H1('Data Exploration For Bills Dataset', style={'textAlign': 'center', 'color': '#0080FF'}),

    #Graph 1
    dcc.Graph(
        id='old_faithful',
        figure={
            'data': [
                go.Scatter(
                    x = df['X'],
                    y = df['Y'],
                    mode = 'markers'
                )
            ],
            'layout': go.Layout(
                title = 'Old Faithful Eruption Intervals v Durations',
                xaxis = {'title': 'Duration of eruption (minutes)'},
                yaxis = {'title': 'Interval to next eruption (minutes)'},
                hovermode='closest'
            )
        }
    ),

    #Graph 2 - Top 5 Service Departments for dataset
    dcc.Graph(
        id='graph-2',
        figure={
            'data': [
                go.Bar(
                    x= graph2_data['counts'],
                    y= graph2_data['unique_values'],
                    orientation='h'
                )],
            'layout': go.Layout(
                title = 'Top 5 Patient Expenditures By Service',
                xaxis = {'title': 'Number of patients'},
                yaxis = {'title': 'Service'},
                hovermode='closest'
            )
        }
    ),

    #Graph 3 - Avg Treatment Cost by Service
    dcc.Graph(
        id='graph-3',
        figure={

            'data': [
                go.Bar(
                    x= graph3_data['Service Department'],
                    y= graph3_data['Treatment Gross']
                )],
            'layout': go.Layout(
                title = 'Patient Spend Breakdown',
                xaxis = {'title': 'Service'},
                yaxis = {'title': 'Cost'},
                hovermode='closest'
            )

        }
    )
    # ,

    #Graph 4 - FRequency of Mammograms done by breast cancer patients
    # dcc.Graph(
    #     id='graph-4',
    #     figure={

    #         'data': [
    #             go.Bar(
    #                 x= graph3_data['Service Department'],
    #                 y= graph3_data['Treatment Gross']
    #             )],
    #         'layout': go.Layout(
    #             title = 'Patient Spend Breakdown',
    #             xaxis = {'title': 'Service'},
    #             yaxis = {'title': 'Cost'},
    #             hovermode='closest'
    #         )

    #     }
    # )


])

#server
if __name__ == '__main__':
    app.run_server(debug=True)