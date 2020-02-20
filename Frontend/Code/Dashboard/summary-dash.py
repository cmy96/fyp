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


#Create df from .csv file
url = 'data/bills1.csv'
url2 = 'data/OldFaithful.csv'
df_bills = pd.read_csv(url, index_col=0)
df_A = pd.read_csv(url2, index_col=0)
# df = pd.read_csv("C:\\Users\\User\\Documents\\fyp\\clinical.csv")
df = pd.read_csv('data/clinical.csv')

# rename columns
death_cause = df['cause_of_death']
death_cause_dict_old = calPercent(df,death_cause,True,"Alive")
death_cause_dict = rename_keys(death_cause_dict_old,\
                        ['Alive', 'Dead- Breast Cancer', 'Dead- Others', 'Dead- Unknown'])

                        
#Data manipulation for graph dataset

##Jess's Graphs
graph2_data = df_bills['Service.Summary..Description'].value_counts(ascending = False).head(5)
graph2_data = graph2_data.rename_axis('unique_values').reset_index(name='counts')

graph3_data = df_bills.groupby('Service.Department.Description')['Gross..exclude.GST.'].sum()
graph3_data = graph3_data.rename_axis('Service Department').reset_index(name='Treatment Gross')
graph3_data = graph3_data.sort_values('Treatment Gross',ascending=False).head(10)



##Tanny's Graphs


#assuming no nan data
TNM = df['TNM_Stage']
TNM_dict = {}
TNM_Stage = TNM.unique()

for stage in TNM_Stage:
    status_dict = {}
    for life_status in death_cause_dict_old.keys():
        
        tmp = df[['TNM_Stage','cause_of_death']]
          
        if life_status == "Alive":
            if len(df[TNM==stage]) > 0 or len(tmp[(tmp['cause_of_death']==life_status) & (tmp['TNM_Stage']==stage)]) > 0:
                NumRecord = len(tmp[(tmp['cause_of_death'].isnull()) & (tmp['TNM_Stage'] == stage)])/len(df[TNM == stage])*100
        else:
            if len(df[TNM==stage]) > 0 or len(tmp[(tmp['cause_of_death']==life_status) & (tmp['TNM_Stage']==stage)]) > 0:
                NumRecord  = len(tmp[(tmp['cause_of_death']==life_status) & (tmp['TNM_Stage']==stage)])/len(df[TNM == stage])*100
            else:
                pass

        status_dict[life_status] = round(NumRecord,2)
    TNM_dict[stage] = status_dict
TNM_dict = dict(sorted(TNM_dict.items(), key=lambda x:operator.getitem(x[1],'breast cancer related')))

# reorganize the previous dict into status for every stage
finalized_dict = {}
for status in death_cause_dict_old.keys():
    for stage in TNM_Stage:
        finalized_dict[status] = [v[status] for k,v in TNM_dict.items()] 

#Binning of diagnosed age - Tanny
age = pd.Series(df['Age_@_Dx'])

bins = np.arange(df['Age_@_Dx'].min(),df['Age_@_Dx'].max() + 4, 4)
df['binned'] = np.searchsorted(bins, df['Age_@_Dx'].values)
age_bin_count = df.groupby(pd.cut(df['Age_@_Dx'], bins=19, precision = 0, right = False)).size()


##Data for Graph 8
graph8_data = df['ER'].value_counts()
graph8_data = graph8_data.rename_axis('type').reset_index(name='counts')

g8_data2 = df['PR'].value_counts()
g8_data2 = g8_data2.rename_axis('type').reset_index(name='counts')


##Styling settings

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

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
                                "float":"right"
                            },
                        )
                    ], className = "one-third column",
                ),
                html.Div(
                    [
                    html.H3('Data Exploration For Breast Cancer Patients Dataset', style={'textAlign': 'left', 'color': '#0080FF', 'margin-top':'30px'})
                    ],
                    className="one-half column", id="title"
                )
        ]),

html.Div([
    html.Div([html.H3("Bills Dataset", style={'textAlign': 'center', 'color': 'red', 'margin-top':'30px'})]
    #add comma and classname col to manipulate
    )
]),


#Jesslyn Charts Wrapper#   
html.Div([
    # #Graph 1
        html.Div([
            dcc.Graph(
            id='old_faithful',
            figure={
                'data': 
                [go.Scatter(
                        x = df_A['X'],
                        y = df_A['Y'],
                        mode = 'markers',
                        marker = dict(color = '#97B2DE'))],
            'layout': go.Layout(
                title = 'Old Faithful Eruption Intervals v Durations',
                xaxis = {'title': 'Duration of eruption (minutes)'},
                yaxis = {'title': 'Interval to next eruption (minutes)'},
                hovermode='closest'
            )
            }   
        )],
        className="pretty_container six columns"
        ),

        html.Div([
            #Graph 2 - Top 5 Service Departments for dataset
            dcc.Graph(
                id='graph-2',
                figure={
                    'data': [
                        go.Bar(
                            x= graph2_data['counts'],
                            y= graph2_data['unique_values'],
                            orientation='h',
                            marker=dict(color=['#97B2DE','#97B2DE','#97B2DE','#97B2DE','#97B2DE'])
                        )],
                    'layout': go.Layout(
                        title = 'Top 5 Patient Expenditures By Service',
                        xaxis = {'title': 'Number of patients'},
                        yaxis = {'title': 'Service'},
                        hovermode='closest'
                    )
                }
            )
        ],
        id="right-column",
        className='six columns'
        ),

        html.Div([
            #Graph 3 - Avg Treatment Cost by Service
            dcc.Graph(
                id='graph-3',
                figure={

                    'data': [
                        go.Bar(
                            x= graph3_data['Service Department'],
                            y= graph3_data['Treatment Gross'],
                            text = graph3_data['Treatment Gross'],
                            marker = dict(color = '#97B2DE')
                            )
                        ],
                    'layout': go.Layout(
                        title = 'Patient Spend Breakdown',
                        xaxis = {'title': 'Service'},
                        yaxis = {'title': 'Cost'},
                        hovermode='closest'
                    )

                }
            )
        ], className="pretty_container eight columns"

    )],className="row flex-display"
),

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
    # ),

html.Div([
    html.Div([html.H3("Clinical Dataset", style={'textAlign': 'center', 'color': 'red', 'margin-top':'30px'})]
    #add comma and classname col to manipulate
    )
]),

##TANNY WRAPPER##
html.Div([
        #Graph 5 - Tanny 1
        html.Div([
            dcc.Graph(
                id='diagnosed-age-histogram',
                figure={
                    'data': [
                        go.Histogram(                
                            x = df['Age_@_Dx'],
                            histnorm='probability',
                            xbins=dict(start=df['Age_@_Dx'].min(), end=df['Age_@_Dx'].max(), size=5),
                            text = list(age_bin_count), 
                            marker = dict(color = '#97B2DE')
                        ),
                    ],
                    'layout': go.Layout(
                        title = "Patient's Diagnosed Age Distribution",
                        xaxis = {'title': 'Diagnosed Age'},
                        yaxis = {'title': 'Percentage of Patients'},

                    )
                }
        )],
        className = "pretty_container six columns"
        ),

        html.Div([
        #Graph 6 - Tanny 2
        dcc.Graph(
            id='Proportion of Patients Alive Vs Dead',
            figure={
                'data': [
                    go.Bar(
                    x=list(death_cause_dict.keys()), 
                    y=list(death_cause_dict.values()),
                    text= list(death_cause_dict.values()),
                    textposition='auto',
                     marker=dict(color=['lightgreen','lightcoral','indianred','lightslategray' ])
            )
                ],
                'layout': go.Layout(
                    title = "Proportion of Patients Alive Vs Dead",
                    xaxis = {'title': 'Cause of Death'},
                    yaxis = {'title': 'Percentage of Patients'},
                    hovermode='closest'
                )
            }
        )],
        className = "pretty_container six columns"
        ),
    html.Div([
        #Graph 7 - Tanny 3
        dcc.Graph(
                id='TNM Stage Alive Vs Dead',
                figure={
                    'data': [
                    go.Bar(
                            x= list(finalized_dict['Alive']),
                            y= list(TNM_dict.keys()),
                            name='Alive',
                            orientation='h',
                            marker=dict(
                            color='lightgreen',
                            line=dict(color='lightgreen', width=3)
                            )
                        ),
                        go.Bar(
                            x= list(finalized_dict['breast cancer related']),
                            y= list(TNM_dict.keys()),
                            name='Dead- Breast cancer related',
                            orientation='h',
                            marker=dict(
                            color='lightcoral',
                            line=dict(color='lightcoral', width=3)
                            )
                        ),
                        go.Bar(
                            x= list(finalized_dict['n']),
                            y= list(TNM_dict.keys()),
                            name='Dead',
                            orientation='h',
                            marker=dict(
                            color='indianred',
                            line=dict(color='indianred', width=3)
                            )
                        ),
                        go.Bar(
                            x= list(finalized_dict['unknown']),
                            y= list(TNM_dict.keys()),
                            name='Unknown',
                            orientation='h',
                            marker=dict(
                            color='lightslategrey',
                            line=dict(color='lightslategrey', width=3)
                            )
                        )
                    ],
                    'layout': go.Layout(
                        title = "TNM Stage Alive Vs Dead",
                        xaxis = {'title': 'Percentage of Patients'},
                        yaxis = {'title': 'Cancer Stages'},
                        hovermode='closest',
                        barmode='stack',

                    )
                }
            )
            ],
            className = "pretty_container five columns"
            ),

    html.Div([
        #Graph 8 - Tanny 4
        dcc.Graph(id="graph-8",
        figure={'data':[
        go.Bar(
            x= list(graph8_data['type']),
            y= list(graph8_data['counts']),
            name='ER',
            marker=dict(color='lightpink')
            # Change labels to percentage ,text=[6,87,68,46]
        ),
        go.Bar(
            x = list(g8_data2['type']),
            y=list(g8_data2['counts']),
            name = 'PR',
            marker = dict(color = 'steelblue')
            #change count to percent
        )
        ],
        'layout': go.Layout(
                        title = "Relationship Between ER & PR",
                        xaxis = {'title': 'Type'},
                        yaxis = {'title': '# of Patients'},
                        hovermode='closest',
                        barmode='group'
        )}
        )
        ],className='pretty_container five columns')
    ],
    className="row flex-display"
    )



],
id="mainContainer",
style={"display": "flex", "flex-direction": "column"}


)

#server
if __name__ == '__main__':
    app.run_server(debug=True, port=8060, dev_tools_hot_reload_max_retry = 10)