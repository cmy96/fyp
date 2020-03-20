"""Create a Dash app within a Flask app."""
from pathlib import Path
import dash
import dash_table
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
from .layout import html_layout
import json
import datetime as dt
import pandas as pd
import pathlib
import plotly.graph_objects as go
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, ClientsideFunction
from textwrap import dedent as d
import numpy as np
import operator
from plotly.subplots import make_subplots
import plotly
import base64
from flask import Blueprint, render_template, request, jsonify, json, redirect, session, url_for
import flask
import logging
from logging.handlers import RotatingFileHandler
from pandas.io.json import json_normalize



#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

external_stylesheets = [
                        'https://fonts.googleapis.com/css?family=Lato',
                        'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
                        'https://codepen.io/chriddyp/pen/bWLwgP.css',

                        dbc.themes.BOOTSTRAP]


#Launch App
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True



################## ORIGINAL DFS FOR RESULTS PAGE + BILLS DASHBOARD ##############################

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



def df_func(output,column1,column2):
    tmp = pd.DataFrame(output.items(), columns=[column1, column2])
    return tmp

#Create df from .csv file
url = 'data/bills1.csv'
url2 = 'data/OldFaithful.csv'
df_bills = pd.read_csv(url, index_col=0)
df_A = pd.read_csv(url2, index_col=0)
# df = pd.read_csv("C:\\Users\\User\\Documents\\fyp\\clinical.csv")
df = pd.read_csv('data/clinical.csv')
patient = pd.read_csv('data/patient1.csv')
#km = pd.read_csv('data/km.csv')


# rename columns
death_cause = df['cause_of_death']
death_cause_dict_old = calPercent(df,death_cause,True,"Alive")
death_cause_dict = rename_keys(death_cause_dict_old,\
                        ['Alive', 'Dead- Breast Cancer', 'Dead- Others', 'Dead- Unknown'])

                        
#Data manipulation for graph dataset

##Jess's Graphs



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

df_age = df['Age_@_Dx']

##Data for Graph 8
ER_dict = {}

ERlist = list(['positive','negative','equivocal','unknown'])
PRlist = list(['positive','negative','equivocal','unknown'])
for status_er in ERlist:
    status_dict = {}
    for status_pr in PRlist:
        tmp = df[['ER','PR']]
        
        if len(df[df.ER==status_er]) > 0 or len(tmp[(tmp['ER']==status_er) & (tmp['PR']==status_pr)]) > 0:
            NumRecord  = len(tmp[(tmp['ER']==status_er) & (tmp['PR']==status_pr)])/len(df[df['ER'] == status_er])*100
        else:
            pass

        status_dict[status_pr] = round(NumRecord,2)
    ER_dict[status_er] = status_dict

ER_dict = dict(sorted(ER_dict.items(),key=lambda i:ERlist.index(i[0])))

er_finalized_dict = {}
for key in ER_dict.keys():
    for value in ER_dict[key].keys():
        er_finalized_dict[key] = [v for k,v in ER_dict[key].items()] 
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

#python codes for survival
s_output = {"6 months before":100.0, "6 months after":96.20, "1 year after":90.10, 
    "2 years after":86.90, "5 years after":80.09, "10 years after":71.22}
df2 = pd.DataFrame(s_output.items(), columns=['Years', 'Survival'])
#python codes for cost
c_output = {"6 months before":3882.80, "6 months after":13112.54, "1 year after":2230.19, 
    "2 years after":1736.58, "5 years after":11800.33, "10 years after":14917.57}
df = pd.DataFrame(c_output.items(), columns=['Years', 'Cost'])


#################################### END OF ORIGINAL DFS FOR RESULTS AND BILLS DASHBOARD ####################################


# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data
clinical = pd.read_csv(DATA_PATH.joinpath("clinical.csv"))
bills = pd.read_csv(DATA_PATH.joinpath("dropped.csv"))

#Get unique vals of each col for filters
tnm_list = clinical["TNM_Stage"].dropna().unique()
ER_list = clinical['ER'].dropna().unique()
pr_list = clinical['PR'].dropna().unique()
Her2_list = clinical['Her2'].dropna().unique()
clinical['Age_@_Dx'] = clinical['Age_@_Dx'].astype(int)





###########################################   Data manipulation for bills charts    ######################################
graph2_data = df_bills['Service.Summary..Description'].value_counts(ascending = False).head(5)
graph2_data = graph2_data.rename_axis('unique_values').reset_index(name='counts')

graph3_data = df_bills.groupby('Service.Department.Description')['Gross..exclude.GST.'].sum()
graph3_data = graph3_data.rename_axis('Service Department').reset_index(name='Treatment Gross')
graph3_data = graph3_data.sort_values('Treatment Gross',ascending=False).head(10)




# gross_list = generate_gross_by_category_data(bills)

gross = bills.groupby(['Consolidated.Main.Group']).sum()
gross_list = {}
for row in gross.itertuples():
    gross_list[row.Index] = row[2]


average = bills.groupby(['Consolidated.Main.Group']).mean()
average_list = {}
for row in average.itertuples():
    average_list[row.Index] = row[1]






###########################################   Data manipulation for clinical charts    ######################################
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
        newdf= df1[columnDF == v]
        dict_list[v] = round(len(newdf)/len(columnDF)*100,2)

    #sort values descendingly
    dict_list = dict(sorted(dict_list.items(),key=operator.itemgetter(1),reverse = True))

    return dict_list

# # this has null
death_cause = clinical['cause_of_death']
death_cause_dict_old = calPercent(clinical,death_cause,True,"Alive")
death_cause_dict = rename_keys(death_cause_dict_old,\
                       death_cause.unique())


#Binning of diagnosed age - Tanny
bins = np.arange(clinical['Age_@_Dx'].min(),clinical['Age_@_Dx'].max() + 4, 4)
clinical['binned'] = np.searchsorted(bins, clinical['Age_@_Dx'].values)
age_bin_count = clinical.groupby(pd.cut(clinical['Age_@_Dx'], bins=19, precision = 0, right = False)).size()


    
def generate_tnm_chart_data(df, dc):
    # for every stage in TNM_stage, it becomes the key for the MAIN dictionary
    #assuming no nan data
    TNM = df['TNM_Stage']
    # TNM_Stage = TNM.dropna().unique()
    TNM_dict={}

    for stage in clinical['TNM_Stage'].dropna().unique():
        status_dict = {}
        # tnm_death_cause_dict = calPercent(df,dc,True,"Alive")

        #every stage in tnm_stage will have a dictionary that holds death_status as key and number of death as value
        for life_status in death_cause_dict_old.keys():
            tmp = df[['TNM_Stage','cause_of_death']]
            condition1 = df[TNM==stage]
            condition2 = tmp[(tmp['cause_of_death']==life_status) & (tmp['TNM_Stage']==stage)]
            condition3 = tmp[(tmp['cause_of_death'].isnull()) & (tmp['TNM_Stage'] == stage)]
            NumRecord = 0

            if life_status == "Alive":
                if len(condition1) > 0 or len(condition2) > 0:
                    NumRecord = len(condition3)/len(condition1)*100
            else:
                if len(condition1) > 0 or len(condition2) > 0:
                    NumRecord  = len(condition2)/len(condition1)*100
                else:
                   NumRecord = 0

            status_dict[life_status] = round(NumRecord,2)
        TNM_dict[stage] = status_dict

    TNM_dict = dict(sorted(TNM_dict.items(), key=lambda x:operator.getitem(x[1],'breast cancer related')))
    
    # reorganize the previous dict into status for every stage
    finalized_dict = {}
    for status in death_cause_dict_old.keys():
        for stage in clinical['TNM_Stage'].dropna().unique():
            finalized_dict[status] = [v[status] for k,v in TNM_dict.items()]
    # print(finalized_dict)
    # print('tnm')
    # print(TNM_dict)
    return finalized_dict


def generate_epr_chart_data(df):
    '''
    :return a dictionary of dictionaries where each ER status has a dictionary of scores (% against a PR status)
    '''
    #all er status aginst each pr values (pos/neg/equivocal/unknown)
    er_dict = {'positive':[], 'negative':[], 'equivocal':[], 'unknown':[]}
    
    #iterate to fill each er status with epr scores
    for key in er_dict:
        status_dict = {'positive': 0.0, 'negative':0.0, 'equivocal':0.0, 'unknown':0.0}
        #Number of ER/PR in given dataset (filtered)
        filtered_er = df[df['ER'] == key]
        if len(filtered_er) > 0:
            num_of_er = len(filtered_er)
            #Calculate pr scores for each er status
            for pr_status in ['positive','negative', 'equivocal','unknown']:
               
                filtered_pr = df[df['PR'] == pr_status]
                if len(filtered_pr) > 0:
                    num_of_epr = len(df[(df['ER'] == key) & (df['PR'] == pr_status)])
                    total_score = num_of_epr/num_of_er*100
                    status_dict[pr_status] = round(total_score,2)
                else:
                    total_score = 0.00
                    status_dict[pr_status] = round(total_score,2)
                    continue
        else:
            # num_of_er = 0.00
            total_score = 0.00
            er_dict[key] = status_dict 
            continue #skip to next er status if er status is not found in df
        
        # er_dict = dict(sorted(er_dict.items(),key=lambda i:ER_list.index(i[0])))
        er_dict[key] = status_dict
    #Convert dictionaries in er_dict to lists
    final = {}
    for item in er_dict.items():
        # print(item)
        final[item[0]] = list(item[1].values())
    # print(final)
    return final 

###########################################################################################################################

def filter_df(df, min1, max1, tnm_select, er_select, pr_select, her2_select):
    '''
        df:dataframe
        age_slider: slider values in a tuple (min, max) - use age_slider[0 or 1] to extract
        er_status: str
        pr_status: str
        her2_status: str

        Purpose of this function filters the full dataset, clinical, using the variables given in the filter panel of the application.

    '''
    condition = (df['Age_@_Dx'] < max1) & (df['Age_@_Dx'] > min1) & (df['ER'] == er_select) & (df['PR'] == pr_select) & (df['TNM_Stage'] == tnm_select) & (df['Her2'] == her2_select)
    output = df[condition]
    return output

def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Clinical Dataset Analysis"),
            html.H3("Welcome to the Summary Dashboard"),
            html.Div(
                id="intro",
                children="The clinical dataset contains past records of breast cancer patients, which doctors can explore with filters for attributes: Age, TNM stage, ER status, PR status and HER2 status.",
            ),
        ],
    )

def generate_controls():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Filter by Age"),
            dcc.RangeSlider(
                id="age_slider",
                min=clinical['Age_@_Dx'].min(),
                max=clinical['Age_@_Dx'].max(),
                value=[30, 65],
                marks={
                    0: '0',
                    20: '20',
                    40: '40',
                    60: '60',
                    80: '80'
                },
                className="dcc_control",
            ),
            html.Br(),
            html.P("Select TNM Stage"),
            dcc.Dropdown(
                id="tnm_select",
                options= [{"label": i, "value": i} for i in tnm_list],
                value=tnm_list[1],
            ),
            html.Br(),
            html.P("Select ER status"),
            dcc.Dropdown(
                id="er_select",
                options=[{"label": i, "value": i} for i in ER_list],
                value=ER_list[0],
            ),
            html.Br(),
            html.P("Select PR status"),
            dcc.Dropdown(
                id="pr_select",
                options=[{"label": i, "value": i} for i in pr_list],
                value=pr_list[0]
            ),
            html.Br(),
            html.P("Select Her2 status"),
            dcc.Dropdown(
                id="her2_select",
                options=[{"label": i, "value": i} for i in Her2_list],
                value=Her2_list[0]
            ),

            html.Br(),
            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            ),
            html.Br(),
        ],
    )


def generate_bills_controls():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.P("Select category"),
            dcc.Dropdown(
                id="category_select",
                options=[{"label": i, "value": i} for i in category_list],
                value=category_list[0]
            ),
            html.Br(),
            html.P("Select category"),
            dcc.Checklist(
                id="category_check",
                options=[{"label": i, "value": i} for i in category_list],
                value=category_list,
                labelStyle={'display': 'block'}
            ),
  
            html.Br(),
            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            ),
            html.Br(),
        ],
    )

layout = dict(
        margin=dict(l=70, b=50, t=50, r=50),
        modebar={"orientation": "v"},
        font=dict(family="Open Sans"),
        xaxis=dict(
            side="top",
            ticks="",
            ticklen=2,
            tickfont=dict(family="sans-serif"),
            tickcolor="#ffffff",
        ),
        yaxis=dict(
            side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" "
        ),
        hovermode="closest",
        showlegend=False,
)

finalized_dict = generate_tnm_chart_data(clinical, death_cause)
er_finalized_dict = generate_epr_chart_data(clinical)

bills_layout = dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    html.Div(      
                        html.H1("Data Exploration For Breast Cancer Patients Dataset")       
                    ), width={"size": 12, "offset": 3},
                ),
            ),

            html.Br(),
            html.Br(),

            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dbc.Container(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div(
                                                dbc.Button(
                                                    "Bills", outline=True, color="secondary", className="mr-1", href="/dashboard/bills", block=True, size="lg", active=True
                                                )
                                            ), width={"size":6}, className="column_padding"
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                dbc.Button(
                                                    "Clinical", outline=True, color="secondary", className="mr-1", href="/dashboard/clinical", block=True, size="lg"
                                                )
                                            ), width={"size":6}
                                        )
                                    ]
                                )
                            )
                        ), width={"size":10, "offset":1}
                    )
                ]
            ),
        ]
    )

clinical_layout = dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    html.Div(      
                        html.H1("Data Exploration For Breast Cancer Patients Dataset")       
                    ), width={"size": 12, "offset": 3},
                ),
            ),

            html.Br(),
            html.Br(),

            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dbc.Container(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div(
                                                dbc.Button(
                                                    "Bills", outline=True, color="secondary", className="mr-1", href="/dashboard/bills", block=True, size="lg"
                                                )
                                            ), width={"size":6}, className="column_padding"
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                dbc.Button(
                                                    "Clinical", outline=True, color="secondary", className="mr-1", href="/dashboard/clinical", block=True, size="lg", active=True
                                                )
                                            ), width={"size":6}
                                        )
                                    ]
                                )
                            )
                        ), width={"size":10, "offset":1}
                    )
                ]
            ),
        ]
    )



gross_list = {'Cancer drugs': 268941532, 'Hospital care': 96600809, 'Investigations': 247463280, 'Other drugs': 37384671, 'Other surgeries': 137540609, 'Others': 67517445, 'Professional services': 87810977, 'Radiation therapy': 78311175, 'Surgical procedures': 53562182}
category_list = list(gross_list.keys())

scatter = {'1': 41, '2': 1651, '3': 844, '4': 576, '5': 214, '6': 801, '7': 2078, '8': 128, '9': 638, '10': 72, '11': 2683, '12': 997, '13': 1093, '14': 1658, '15': 970, '16': 17209, '17': 295, '18': 5706, '0001a176c3bec5cdd819a6': 2739, '20': 3051}

bills_dashboard = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H2("Filters"),
                                    className="cardheader_layout"
                                ),
                                dbc.CardBody(
                                    [
                                        generate_bills_controls()
                                    ]
                                )
                            ], className="filter_layout", id="filter_box"
                        )
                    ), width = {"size":3},
                ),

                dbc.Col(
                    html.Div(
                        [
                            dbc.Container(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(

                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            html.H2("Bills Dataset"),
                                                        ),
                                                        dbc.CardBody(
                                                            [
                                                                dbc.Container(
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                #Graph 1 - Top 5 Service Departments for dataset
                                                                                dcc.Graph(
                                                                                    id='expenditure-scatter',
                                                                                        figure={
                                                                                            'data':
                                                                                                [
                                                                                                    go.Scatter(                
                                                                                                        x=list(scatter.keys()),
                                                                                                        y=list(scatter.values()),
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
                                                                                            },                
                                                                                    )
                                                                                ),
                                                                            dbc.Col(
                                                                                dcc.Graph(
                                                                                    id='gross_service',
                                                                                    figure={
                                                                                        'data': [
                                                                                            go.Pie(
                                                                                                labels= list(gross_list.keys()),
                                                                                                values= list(gross_list.values()),
                                                                                                #labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen'],
                                                                                                #values = [4500, 2500, 1053, 500],
                                                                                                #marker = dict( line=dict(color='#000000', width=2))
                                                                                            )
                                                                                        ],
                                                                                        'layout': go.Layout(
                                                                                            title='Gross Expenditure by Category',
                                                                                            #xaxis = {'title': "Patient's Medical Service Types", 'automargin': True},
                                                                                            #yaxis = {'title': 'Total Cost ($)'},
                                                                                            #width = 500,
                                                                                            #height = 530,
                                                                                            hovermode='closest'
                                                                                        ),
                                                                                        
                                                                                    }
                                                                                )
                                                                            ),

                                                                            dbc.Col(
                                                                                dcc.Graph(
                                                                                    id='average_service',
                                                                                    figure={
                                                                                        'data': [
                                                                                            go.Bar(
                                                                                                x= list(average_list.keys()),
                                                                                                y= list(average_list.values()),
                                                                                                #labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen'],
                                                                                                #values = [4500, 2500, 1053, 500],
                                                                                                #marker = dict( line=dict(color='#000000', width=2))
                                                                                            )
                                                                                        ],
                                                                                        'layout': go.Layout(
                                                                                            title='Average Expenditure by Category',
                                                                                            #xaxis = {'title': "Patient's Medical Service Types", 'automargin': True},
                                                                                            #yaxis = {'title': 'Total Cost ($)'},
                                                                                            #width = 500,
                                                                                            #height = 530,
                                                                                            hovermode='closest'
                                                                                        ),
                                                                                        
                                                                                    }
                                                                                ),
                                                                            ),

                                                                            dbc.Col(
                                                                                average_list
                                                                            )
                                                                        ]
                                                                    )
                                                                )
                                                            ]
                                                        )
                                                    ], id="chart_box"
                                                ),
                                            ),
                                        ]
                                   ),                
                                ]
                            )
                 
                        ]
                    ), width = {"size":9},
                )
            ]
        )
    ]
)


clinical_dashboard = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H2("Filters"),
                                    className="cardheader_layout"
                                ),
                                dbc.CardBody(
                                    [
                                        generate_controls()
                                    ]
                                )
                            ], className="filter_layout"
                        )
                    ), width = {"size":3},
                ),

                dbc.Col(
                    html.Div(
                        [
                            dbc.Container(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(

                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            html.H2("Clinical Dataset"),
                                                        ),
                                                        dbc.CardBody(
                                                            [
                                                                dbc.Container(
                                                                    [
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Col(
                                                                                    dcc.Graph(
                                                                                        id="age-distribution-hist",
                                                                                        figure={
                                                                                            'data':[
                                                                                                {
                                                                                                'x': clinical['Age_@_Dx'],
                                                                                                'name': 'Age Distribution',
                                                                                                'type': 'histogram',
                                                                                                'autobinx': False,
                                                                                                'histnorm':'probability',
                                                                                                'xbins': {
                                                                                                    'start': clinical['Age_@_Dx'].min(),
                                                                                                    'end': clinical['Age_@_Dx'].max(),
                                                                                                    'size': 5
                                                                                                }
                                                                                                }
                                                                                            ],
                                                                                            'layout':go.Layout(
                                                                                                title = "Patient's Diagnosed Age Distribution",
                                                                                                xaxis = {'title': 'Diagnosed Age'},
                                                                                                yaxis = {'title': 'Percentage of Patients'},

                                                                                            )
                                                                                        }
                                                                                    ),           
                                                                                ),

                                                                                dbc.Col(
                                                                                    dcc.Graph(
                                                                                        id="alive_dead_bar",
                                                                                        figure ={
                                                                                            'data': 
                                                                                            [
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
                                                                                    )
                                                                                )
                                                                            ]
                                                                        ),

                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Col(
                                                                                    dcc.Graph(
                                                                                        id="tnm-stage-stacked-bar",
                                                                                        figure={
                                                                                            'data': [
                                                                                            go.Bar(
                                                                                                    x= finalized_dict['Alive'],
                                                                                                    y= clinical['TNM_Stage'].dropna().unique(),
                                                                                                    name='Alive',
                                                                                                    orientation='h',
                                                                                                    marker=dict(
                                                                                                    color='lightgreen',
                                                                                                    line=dict(color='lightgreen', width=3)
                                                                                                    )
                                                                                                ),
                                                                                                go.Bar(
                                                                                                    x= finalized_dict['breast cancer related'],
                                                                                                    y= clinical['TNM_Stage'].dropna().unique(),
                                                                                                    name='Dead- Breast cancer related',
                                                                                                    orientation='h',
                                                                                                    marker=dict(
                                                                                                    color='lightcoral',
                                                                                                    line=dict(color='lightcoral', width=3)
                                                                                                    )
                                                                                                ),
                                                                                                go.Bar(
                                                                                                    x=finalized_dict['n'],
                                                                                                    y= clinical['TNM_Stage'].dropna().unique(),
                                                                                                    name='Dead',
                                                                                                    orientation='h',
                                                                                                    marker=dict(
                                                                                                    color='indianred',
                                                                                                    line=dict(color='indianred', width=3)
                                                                                                    )
                                                                                                ),
                                                                                                go.Bar(
                                                                                                    x= finalized_dict['unknown'],
                                                                                                    y= clinical['TNM_Stage'].dropna().unique(),
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
                                                                                                showlegend=False
                                                                                            )
                                                                                        }
                                                                                    ),
                                                                                ),
                                                                                dbc.Col(
                                                                                    dcc.Graph(
                                                                                        id='er_pr_chart',
                                                                                        figure={
                                                                                            'data': [
                                                                                                go.Bar(
                                                                                                    x= [er_finalized_dict['positive'][0],er_finalized_dict['negative'][0],er_finalized_dict['equivocal'][0]],
                                                                                                    y= ['positive','negative','equivocal','unknown'],
                                                                                                    name='PR Positive',
                                                                                                    orientation='h',
                                                                                                    marker=dict(
                                                                                                    color='palegreen',
                                                                                                    line=dict(color='palegreen', width=3)
                                                                                                    )
                                                                                                ),
                                                                                                go.Bar(
                                                                                                    x= [er_finalized_dict['positive'][1],er_finalized_dict['negative'][1],er_finalized_dict['equivocal'][1]],
                                                                                                    y= ['positive','negative','equivocal','unknown'],
                                                                                                    name='PR Negative',
                                                                                                    orientation='h',
                                                                                                    marker=dict(
                                                                                                    color='lightpink',
                                                                                                    line=dict(color='lightpink', width=3)
                                                                                                    )
                                                                                                ),
                                                                                                go.Bar(
                                                                                                    x= [er_finalized_dict['positive'][2],er_finalized_dict['negative'][2],er_finalized_dict['equivocal'][2]],
                                                                                                    y= ['positive','negative','equivocal','unknown'],
                                                                                                    name='PR Equivocal',
                                                                                                    orientation='h',
                                                                                                    marker=dict(
                                                                                                    color='lightblue',
                                                                                                    line=dict(color='lightblue', width=3)
                                                                                                    )
                                                                                                ),
                                                                                                go.Bar(
                                                                                                    x= [er_finalized_dict['positive'][3],er_finalized_dict['negative'][3],er_finalized_dict['equivocal'][3]],
                                                                                                    y= ['positive','negative','equivocal','unknown'],
                                                                                                    name='PR Unknown',
                                                                                                    orientation='h',
                                                                                                    marker=dict(
                                                                                                    color='lightgrey',
                                                                                                    line=dict(color='lightgrey', width=3)
                                                                                                    )
                                                                                                )
                                                                                            ],
                                                                                            'layout': go.Layout(
                                                                                                title = "Relationship between ER & PR",
                                                                                                xaxis = {'title': 'Percentage of ER & PR'},
                                                                                                yaxis = {'title': 'ER/PR stages'},
                                                                                                hovermode='closest',
                                                                                                barmode='stack',
                                                                                            )
                                                                                        }
                                                                                    ),
                                                                                )
                                                                            ]
                                                                        )
                                                                    ]
                                                                )
                                                                                                
                                                            ]
                                                        )
                                                    ]
                                                ),
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ]
                    ), width = {"size":9}                           
                )
            ]
        )
    ]
)
                                        
                                                                


Doctor_View = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is Doctor Tab!", className="card-text"),
        ]
    ),
    className="mt-3",
)

Patient_View = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is Patient Tab!", className="card-text"),
        ]
    ),
    className="mt-3",
)


card = html.Div(
    [

        dbc.Row(
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            label="Doctor View", tab_id="Doctor"
                                        ),
                                        dbc.Tab(label="Patient View", tab_id="View"),
                                    ],
                                    id="card-tabs",
                                    card=True,
                                    active_tab="Doctor",
                                )
                            ),
                            dbc.CardBody(html.P(id="card-content", className="card-text")),
                        ]
                    )
                ], width=10,
            ), justify="center",
        )
    ]
)

survival_layout = dbc.Container(
    [        
        dbc.Row(
             #dbc.Col(
                [
                    html.H1("Survival Prediction")
                ], align="center", justify="center",
            ),   
            
        #),
        html.Br(),
        html.Br(),

        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dbc.Button(
                                "Survival Prediction", outline=True, color="secondary", className="mr-1", href="/survival/", block=True, size="lg", active=True
                            ),
                        ]
                    ), width = {"size":4, "offset":2},
                ),

                dbc.Col(
                    html.Div(
                    [
                        dbc.Button(
                            "Cost Prediction", outline=True, color="secondary", className="mr-1", href="/cost/", block=True, size="lg"
                        ), 
                    ]
                    ),width = {"size":4}
                )
            ]
        )
    ] 
)

#cost and survival outputs
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

cost_layout = dbc.Container(
    [

        dbc.Row(
            #dbc.Col(
            [
                html.H1("Cost Prediction")
            ], align="center", justify="center",
        ),  

        html.Br(),
        html.Br(),

        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                    [
                        dbc.Button(
                            "Survival Prediction", outline=True, color="secondary", className="md-4", href="/survival/", block=True, size="lg"
                        ),
                    ]
                    ), width = {"size":4, "offset": 2}
                ),

                dbc.Col(
                    html.Div(
                    [
                          dbc.Button(
                            "Cost Prediction", outline=True, color="secondary", className="md-4", href="/cost/", block=True, size="lg", active=True
                        ),
                    ]
                    ), width = {"size":4}
                )
            ]
        )
    ]
)
#cost graphs for FC
cost_graphs =  html.Div(
    [
        
        dbc.Row(
            dbc.Col(
                
                html.Div(
                [
                    dcc.Graph(
                        id='Cost Table',
                        figure={
                        'data': [
                            go.Table(
                                header=dict(values=["<b>Years</b>","<b>Cost($)</b>"],
                                fill_color='paleturquoise',
                                align='center'),
                                cells=dict(
                                    values=[df.Years, df.Cost],
                                    fill_color='white',
                                    align='center'
                                )
                            )
                        ],
                        'layout':go.Layout(
                            title="Table",
                            height=400,
                            width=1200,
                        )
                    }), 
                    ],
                ), width = {"size":4, "offset": 2}
                
            ),
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                [ 
                    #cost bar chart
                    dcc.Graph(
                        id="Cost Prediction",
                        figure=go.Figure(
                            data=[trace1,trace2],
                            layout=go.Layout(
                                title="Patient's Cost Prediction ($)",
                                height =400,
                                width=1200,

                        
                            )
                        ),
                    )
                ],
                ), width = {"size":4, "offset": 2}
            )
        )
    ]
)
#patient's graphs
# patient_graphs =  html.Div(
#     [

#         dbc.Row(
#             dbc.Col(
#                 html.Div(
#                 [
#                     dcc.Graph(
#                         id='Survivability Rate Table',
#                         figure={
#                         'data': [
#                         go.Table(
#                             header=dict(values=["<b>Years</b>","<b>Survivability Rate</b>"],
#                             fill_color='paleturquoise',
#                             align='center'),
#                             cells=dict(values=[df2.Years, df2.Survival],
#                             fill_color='white',
#                             align='center')
#                         )],
#                         'layout': go.Layout(
#                             height=400,
#                             width=1200,
#                         )


#                 }),
#                 ], 
#             ),  width={"offset":2},
#         ),
#         ),
#         dbc.Row(
#             dbc.Col(
#                 html.Div(
#                 [
#                     dcc.Graph(
#                         id="Survival Prediction",
#                         figure=go.Figure(
#                             data=[trace1s,trace2s],
#                         layout=go.Layout(
#                             title="Patient's Survival Rates Prediction (%)"
#                         ),
#                         ),
#                     ),
#                 ], 
#                 ),  width={"offset":2},
#             ),
#         ),
#         dbc.Row(
#             dbc.Col(
#                 html.Div(
#                 [ 
#                     html.Img(
#                         src=app.get_asset_url('waffle1.png'),
#                         id="waffle-1",
#                         style={
#                             "height": "400px",
#                             "width": "1200px",
#                             "margin-bottom": "25px",
#                             "margin-left":"px"
#                         },
#                     ),
                    
#                     html.P('Out of a 100 random women, 5 will be dead within 10 years... '),
#                     html.Br(),
#                     html.Img(
#                         src=app.get_asset_url('waffle2.png'),
#                         id="waffle-2",
#                         style={
#                             "height": "400px",
#                             "width": "1200px",
#                             "margin-bottom": "25px",
#                             "margin-left":"px"
#                         },
#                     ),
#                     html.Div([
#                         html.P("Out of 100 random breast cancer patients, 95 will survive within the 10 year time period.")
#                     ]), 
                    
#                 ], 
#                 ),  width={"offset":2},
#             ), 
#         ),        
#     ]
# )


patient_graphs = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H2("Chart Name"),
                                    className="cardheader_layout"
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id='Survivability Rate Table',
                                            figure={
                                            'data': [
                                                go.Table(
                                                    columnwidth = [300,300],
                                                    header=dict(values=["<b>Years</b>","<b>Survivability Rate</b>"],
                                                    fill_color='paleturquoise',
                                                    align='center'),
                                                    cells=dict(
                                                        values=[df2.Years, df2.Survival],
                                                        fill_color='white',
                                                        align='center',
                                                        font_size=20,
                                                        height=30
                                                    )
                                                )

                                                ]
                                                
                                            }
                                        )
                                    ]
                                )
                            ]
                        )
                    ), width={"size": 5, "offset":1}
                ),
                dbc.Col(
                    html.Div(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H2("Chart Name"),
                                    className="",
                                ),
                                dbc.CardBody(
                                    [
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
                            ]
                        )
                    ), width={"size":5}
                )

            ]
        ),
        
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H2("Chart Name"),
                                    className="",
                                ),
                                dbc.CardBody(
                                    [
                                        html.Img(
                                            src=app.get_asset_url('waffle1.png'),
                                            id="waffle-1",
                                            style={
                                                "height": "200px",
                                                "width": "600px",
                                                "margin-bottom": "25px",
                                                "margin-left":"px"
                                            },
                                        ),
                                        html.P('Out of a 100 random women, 5 will be dead within 10 years... '),
                                    ]
                                )
                            ]
                        )
                    ), width={"size":5, "offset":1}
                ),
                dbc.Col(
                    html.Div(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H2("Chart Name"),
                                    className="",

                                ),
                                dbc.CardBody(
                                    [
                                        html.Img(
                                            src=app.get_asset_url('waffle2.png'),
                                            id="waffle-2",
                                            style={
                                                "height": "200px",
                                                "width": "600px",
                                                "margin-bottom": "25px",
                                                "margin-left":"px"
                                            },
                                        )
                        
                                    ]                            
                                )
                            ]
                        )
                    ), width={"size":5}
                )
            ]
        )
    ]
)



x = patient["x"]
y = patient["y"]


surv4 = (go.Scatter(x=x//365.25, y=y, name="hv",
                    line_shape='hv'))


# x = km["time"]
# y = km["estimate"]
# lower = km["lower"]
# upper = km["upper"]
# km_upper = go.Scatter(x=x, y=y,
#     fill=None,
#     mode='lines',
#     line_color='indigo',
#     name='Fair',
# )

# km_lower = go.Scatter( x=x,
#     y=upper,
#     fill='tonexty', # fill area between trace0 and trace1
#     mode='lines', 
#     line_color='lightblue',
#     name='Ideal',
# )
# km = go.Scatter( x=x,
#     y=lower,
#     fill='tonexty', # fill area between trace0 and trace1
#     mode='lines', 
#     line_color='lightblue',
#     name='Ideal',
# )

doctor_graphs = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H2("Chart Name"),
                                    className="cardheader_layout"
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="Survival Prediction",
                                            figure=go.Figure(
                                                                data=[surv4],
                                                                layout=go.Layout(
                                                                    title="Patient's Predicted Survival Chart",
                                                                    #height=400,
                                                                    #width=800,
                                                                ),
                                                            ),
                                        )
                                    ]
                                )
                            ]
                        )
                    ), width={"size":10, "offset":1}
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H2("Chart Name"),
                                    className="cardheader_layout"
                                ),
                                dbc.CardBody(
                                    [
                                        # dcc.Graph(
                                        # id="Survival Prediction",
                                        # figure=go.Figure(
                                        #                     data=[km_upper,km_lower,km],
                                        #                     layout=go.Layout(
                                        #                         title="Patient's Predicted Survival Chart",
                                        #                         height=400,
                                        #                         width=1200,  
                                        #                         xaxis_range=(0, 10),
                                        #                         hovermode= "closest",
                                        #                     ),
                                        #                 ),
                                        # ),
                                    ]
                                )
                            ]
                        )
                    ), width={"size":5, "offset":1}
                
                ),

                dbc.Col(
                    html.Div(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H2("Chart Name"),
                                    className="cardheader_layout"
                                ),
                                dbc.CardBody(
                                    [
                                        # dcc.Graph(
                                        # id="Survival Prediction",
                                        # figure=go.Figure(
                                        #                     data=[km_upper,km_lower,km],
                                        #                     layout=go.Layout(
                                        #                         title="Patient's Predicted Survival Chart",
                                        #                         height=400,
                                        #                         width=1200,  
                                        #                         xaxis_range=(0, 10),
                                        #                         hovermode= "closest",
                                        #                     ),
                                        #                 ),
                                        # ),
                                    ]
                                )
                            ]
                        )
                    ), width={"size":5, "offset":0}
                ),
            ]
        ),

                dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H2("Chart Name"),
                                    className="cardheader_layout"
                                ),
                                dbc.CardBody(
                                    [
                                        # dcc.Graph(
                                        # id="Survival Prediction",
                                        # figure=go.Figure(
                                        #                     data=[km_upper,km_lower,km],
                                        #                     layout=go.Layout(
                                        #                         title="Patient's Predicted Survival Chart",
                                        #                         height=400,
                                        #                         width=1200,  
                                        #                         xaxis_range=(0, 10),
                                        #                         hovermode= "closest",
                                        #                     ),
                                        #                 ),
                                        # ),
                                    ]
                                )
                            ]
                        )
                    ), width={"size":5, "offset":1}
                
                ),

                dbc.Col(
                    html.Div(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H2("Chart Name"),
                                    className="cardheader_layout"
                                ),
                                dbc.CardBody(
                                    [
                                        # dcc.Graph(
                                        # id="Survival Prediction",
                                        # figure=go.Figure(
                                        #                     data=[km_upper,km_lower,km],
                                        #                     layout=go.Layout(
                                        #                         title="Patient's Predicted Survival Chart",
                                        #                         height=400,
                                        #                         width=1200,  
                                        #                         xaxis_range=(0, 10),
                                        #                         hovermode= "closest",
                                        #                     ),
                                        #                 ),
                                        # ),
                                    ]
                                )
                            ]
                        )
                    ), width={"size":5, "offset":0}
                ),
            ]
        ),
   
    ]
)


doctor_items = [
    dbc.DropdownMenuItem("Doctor View", href="/survival/doctor", active=True),
    dbc.DropdownMenuItem("Patient View", href="/survival/patient"),
]

patient_items = [
    dbc.DropdownMenuItem("Doctor View", href="/survival/doctor"),
    dbc.DropdownMenuItem("Patient View", href="/survival/patient", active=True),
]


jumbotron =  '''
    <div class="jumbotron jumbo">
        <h1 class="display-4">Filters</h1>
        <p><h4>Filter by:</h4></p>
        </br>
        <p><h6>Service</h6></p>
    </div>
    '''




doctor_button = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                    [
                        html.Br(),
                        html.Br(),
                        html.Div(
                            [
                                dbc.DropdownMenu(
                                    doctor_items, label="Doctor View", color="primary", className="md-4", bs_size="lg"
                                ),


                            ]
                        )
                    ]
                ), width = {"size":4, "offset": 2}
                )
            ]
        )
    ]
)

patient_button = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                    [
                        html.Br(),
                        html.Br(),
                        html.Div(
                            [
                                dbc.DropdownMenu(
                                    patient_items, label="Patient View", color="primary", className="md-4", bs_size="lg"
                                ),


                            ],
                        )
                    ]
                    ), width = {"size":4, "offset": 2} 
                )
            ]
        )
    ]
)

def Add_Dash(server):
    """Create a Dash app."""
    external_stylesheets = ['/static/dist/css/styles.css',
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

    dash_app.config.suppress_callback_exceptions = True

    # Override the underlying HTML template. File in layout.py
    dash_app.index_string = html_layout


    #Create Dash Layout 
    dash_app.layout = html.Div([

        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content'),
        html.Div(id='output-graph'),
        html.Div(id='input'),
        html.Div(id='output'),
        #dcc.Graph(id='graph-with-slider'),
    

    ])

    #Initialize callbacks after our app is loaded
    # Pass dash_app as a parameter
    init_callbacks(dash_app)

    return dash_app.server



def init_callbacks(dash_app):
    @dash_app.callback(
        
            dash.dependencies.Output('page-content', 'children'),                              
        [
            dash.dependencies.Input('url', 'pathname'), 
        ]
    )
                       
    def display_page(pathname):        
        if pathname =="/dashboard/bills":
            return bills_layout, bills_dashboard
        elif pathname =="/dashboard/clinical":
            return clinical_layout, clinical_dashboard
        elif pathname == "/results/":
            cookies = session['received']
            cookies = str(cookies, 'utf-8')
            cookies =cookies.split(",")
            print(cookies)
            cookie = html.H1(cookies)

            #ok = json_normalize(cookies)
            #ok = pd.read_json(cookies)
            ok = pd.DataFrame(cookies)

            something =   dcc.Graph(
                        id='Cost Table',
                        figure={
                        'data': [
                            go.Table(
                                header=dict(values=["<b>Years</b>","<b>Cost($)</b>"],
                                fill_color='paleturquoise',
                                align='center'),
                                cells=dict(
                                    values=[ok.iloc[0], ok.iloc[1]],
                                    fill_color='white',
                                    align='center'
                                )
                            )
                        ],
                        'layout':go.Layout(
                            title="Table",
                            height=400,
                            width=1200,
                        )
                    })



            #return survival_layout, doctor_button, doctor_graphs
            return something
        elif pathname == "/survival/":
            return survival_layout, doctor_button, doctor_graphs
        elif pathname == "/cost/":
            return cost_layout, cost_graphs
        elif pathname =="/survival/patient":
            return survival_layout, patient_button, patient_graphs
        elif pathname =="/survival/doctor":
            return survival_layout, doctor_button, doctor_graphs

    @dash_app.callback(
        [
            dash.dependencies.Output('age-distribution-hist', 'figure'),
            dash.dependencies.Output('alive_dead_bar','figure'),
            dash.dependencies.Output('tnm-stage-stacked-bar', 'figure'),
            dash.dependencies.Output('er_pr_chart','figure'),
        ],
        [
            dash.dependencies.Input('age_slider', 'value'),
            dash.dependencies.Input('tnm_select','value'),
            dash.dependencies.Input('er_select','value'),
            dash.dependencies.Input('pr_select','value'),
            dash.dependencies.Input('her2_select','value'),
        ]
    )

    
    def update_all(age_slider,tnm_select, er_select, pr_select, her2_select):
        #Slice Df according to inputs in filter
        df = filter_df(clinical, age_slider[0], age_slider[1] , tnm_select, er_select, pr_select, her2_select)
        # df = clinical[(clinical['Age_@_Dx'] > age_slider[0]) & (clinical['Age_@_Dx'] < age_slider[1]) & (clinical['ER'] == er_select) ]

        #editing alive vs dead bar chart - overwrite initial version
        cause = df['cause_of_death']
        dcdict_new = calPercent(df,cause,True,"Alive")
        # print(calPercent(df, df['cause_of_death'], True, "Alive"))
        dcdict = rename_keys(dcdict_new,\
                                ['Alive', 'Dead- Breast Cancer', 'Dead- Others', 'Dead- Unknown'])

        #Overwrite initial tnm chart
        fdict = generate_tnm_chart_data(df, cause)

        #Overwrite original chart data with sliced dataset according to filters
        er_dict = generate_epr_chart_data(df)
        # print(er_dict)
        
        figure={
            'data': 
            [
                {
                    'x': df['Age_@_Dx'],
                    'name': 'Age Distribution',
                    'type': 'histogram',
                    'autobinx': False,
                    'histnorm':'probability',
                    'xbins': 
                    {
                        'start': df['Age_@_Dx'].min(),
                        'end': df['Age_@_Dx'].max(),
                        'size': 5
                    }
                }
            ]
        }

        figure2 ={
            'data':
            [
                {
                    'x': list(dcdict.keys()), 
                    'y':list(dcdict.values()), 
                    'type':'bar', 
                    'text':list(dcdict.values()), 
                    'textposition':'auto', 
                    'marker': dict(color=['lightgreen','lightcoral','indianred','lightslategray' ])
                }
            ]
        }

        figure3 ={
                'data': [
                        dict(
                            x= fdict['Alive'],
                            y= clinical['TNM_Stage'].dropna().unique(),
                            type='bar',
                            name='Alive',
                            orientation='h',
                            marker=dict(
                            color='lightgreen',
                            line=dict(color='lightgreen', width=3))
                        ),
                        dict(
                            x= fdict['breast cancer related'],
                            y= clinical['TNM_Stage'].dropna().unique(),
                            type='bar',
                            name='Dead- Breast cancer related',
                            orientation='h',
                            marker=dict(
                            color='lightcoral',
                            line=dict(color='lightcoral', width=3)
                            )
                        ),
                        dict(
                            x= fdict['n'],
                            y= clinical['TNM_Stage'].dropna().unique(),
                            type='bar',
                            name='Dead',
                            orientation='h',
                            marker=dict(
                            color='indianred',
                            line=dict(color='indianred', width=3)
                            )
                        ),
                        dict(
                            x= fdict['unknown'],
                            y= clinical['TNM_Stage'].dropna().unique(),
                            type='bar',
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
        figure4 = {
                    'data': [
                            dict(
                                x= [er_dict['positive'][0],er_dict['negative'][0],er_dict['equivocal'][0]],
                                y= ER_list,
                                type='bar',
                                name='PR Positive',
                                orientation='h',
                                marker=dict(
                                color='palegreen',
                                line=dict(color='palegreen', width=3)
                                )
                            ),
                            dict(
                                x= [er_dict['positive'][1],er_dict['negative'][1],er_dict['equivocal'][1]],
                                y= ER_list,
                                type='bar',
                                name='PR Negative',
                                orientation='h',
                                marker=dict(
                                color='lightpink',
                                line=dict(color='lightpink', width=3)
                                )
                            ),
                            dict(
                                x= [er_dict['positive'][2],er_dict['negative'][2],er_dict['equivocal'][2]],
                                y= ER_list,
                                type='bar',
                                name='PR Equivocal',
                                orientation='h',
                                marker=dict(
                                color='lightblue',
                                line=dict(color='lightblue', width=3)
                                )
                            ),
                            dict(
                                x= [er_dict['positive'][3],er_dict['negative'][3],er_dict['equivocal'][3]],
                                y= ER_list,
                                type='bar',
                                name='PR Unknown',
                                orientation='h',
                                marker=dict(
                                color='lightgrey',
                                line=dict(color='lightgrey', width=3)
                                )
                            )
                        ],
                        'layout': dict(
                            title = "Relationship between ER & PR",
                            xaxis = {'title': 'Percentage of ER & PR'},
                            yaxis = {'title': 'ER/PR Status'},
                            hovermode='closest',
                            barmode='stack'
                        )
                    }
        return figure, figure2, figure3, figure4


        @dash_app.callback(
            [

                dash.dependencies.Output('gross_service','figure'),
            ],
            [

                dash.dependencies.Input('category_select','value'),
            ]
        )

        def update_bills(category_select):
            updated_gross = {}
            for i in category_select:
                updated_gross[i] = gross_list[i]

            figure5={
                'data': [
                    go.Pie(
                        labels= list(updated_gross[category_select].keys()),
                        values= list(updated_gross[category_select].values()),
                    )
                ],
                'layout': go.Layout(
                    title='Gross Expenditure by Category',

                    hovermode='closest'
                ),
                                                                                        
            }

            return figure5
            
            




