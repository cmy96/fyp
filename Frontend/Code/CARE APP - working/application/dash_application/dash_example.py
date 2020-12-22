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

import kaplan_meier2 as KM2

import base64
from flask import Blueprint, render_template, request, jsonify, json, redirect, session, url_for
import flask
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from matplotlib import cm
from pywaffle import Waffle
import math
import os, shutil
from plotly.tools import mpl_to_plotly
import dash_table


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

filename = 'application\\assets\\waffle-chart-km.png' #where picture will be stored, replace w ur own
if os.path.exists(filename):
    os.remove(filename) #remove old

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data
clinical = pd.read_csv(DATA_PATH.joinpath("clinical_full_data.csv"))
bills2 = pd.read_csv(DATA_PATH.joinpath("dropped.csv"))
bills = pd.read_csv("data/bills.csv")

listToDrop = ['NRIC','dob','Has Bills?','Side','Hospital','KKH','NCCS','SGH','END_OF_ENTRY']
input_df = KM2.kaplan_meier_load_clinical_df(listToDrop)

#Set Variables
tnm_list = clinical["Stage"].dropna().unique()
ER_list = clinical['ER'].dropna().unique()
pr_list = clinical['PR'].dropna().unique()
Her2_list = clinical['Her2'].dropna().unique()

T_list = clinical['T'].unique()
N_list = clinical['N'].unique()
M_list = clinical['M'].unique()

TNM_options_dict = {'T': T_list, 'N': [], 'M': []}


clinical.dropna(axis=0,subset=['Age_@_Dx'],inplace=True)
clinical['Age_@_Dx'] = clinical['Age_@_Dx'].astype(int)
max_patient_num = len(bills2['Case.No'])
jumbotron =  '''
    <div class="jumbotron jumbo">
        <h1 class="display-4">Filters</h1>
        <p><h4>Filter by:</h4></p>
        </br>
        <p><h6>Service</h6></p>
    </div>
    '''
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


###########################################   Prediction model visualisations    ######################################

# Waffle Chart is the chart with stickman icons found in Survival Prediction tab 
# Home > Prediction Input Page > Survival Prediction tab

def generate_waffle_chart():
    '''
        parameters: waffle2data should be the output of survival model
        return: waffle chart image in assets folder (for results page to extract)
    '''
    #set the vars needed
    populate_this_dict = {'0.5 years': 0, '1 years': 0, '2 years': 0, '5 years':0, '10 years':0} #this is done to set the initial dict. Populate w the actual values to pass the chart printer.
    list_of_dict_keys = list(populate_this_dict.keys()) #retrieve all the keys, convert to list
    to_sum = []

    waffle2data = pd.read_csv("..\\middleWomen\\survival.csv") #extract model predictions survival csv
    waffle2data.drop("Unnamed: 0", axis=1, inplace=True) #drop extra col
    print("WAFFLE DATA")
    print(waffle2data)

    #Arithmetics - still need old cols, replace old cols at end.
    # all values in waffle2data(survival rate in deci) * 100 to convert to % 
    # Here we will take prev num minus next num and absolute it
    # find out who is left out of the original pool of 100%
    for i in range(len(list_of_dict_keys)):
        if i == 0:
            populate_this_dict[list_of_dict_keys[i]] = int(abs(100 - (waffle2data[list_of_dict_keys[i]]*100)))
            to_sum.append(populate_this_dict[list_of_dict_keys[i]])
        elif i < len(list_of_dict_keys)-1:
            populate_this_dict[list_of_dict_keys[i]] = int(abs((waffle2data[list_of_dict_keys[i-1]]*100)- (waffle2data[list_of_dict_keys[i]] * 100)))
            to_sum.append(populate_this_dict[list_of_dict_keys[i]])
        elif i == len(list_of_dict_keys)-1:
            survived = 100 - sum(to_sum) #100% - sum of all values in dict, note tt current value of survived10yrs = 0
            populate_this_dict[list_of_dict_keys[i]] = survived
        else:
            print('i is an invalid value')
        
    print("this will be passed to the chart gen", populate_this_dict)
    populate_this_dict = rename_keys(populate_this_dict, ['Dead - in 6 months', 'Dead - 1 year', 'Dead - 2 years', 'Dead - 5 years', 'Survived - 10 years'])
    output = populate_this_dict

    plt.figure(
            FigureClass=Waffle, 
            rows=5, 
            values=output, 
            colors= ['#800000', '#FF0000', '#F08080', '#FFA07A', '#3CB371'],
            legend={
                'labels': ["{0} ({1})".format(k, v) for k, v in output.items()],
                'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
            icons='child', icon_size=14, 
            icon_legend=True,
            figsize=(10, 9)
            ,
            title={
            'label': 'Survival Rates for Breast Cancer Patients ',
            'loc': 'center',
            'fontdict':{'fontsize':8
            }}
    )
    filename = 'application\\assets\\waffle-chart-survived.png' #where picture will be stored
    if os.path.exists(filename):
        os.remove(filename) #remove old
    plt.savefig(filename ,bbox_inches='tight', pad_inches=0) #replace 



###########################################   Data manipulation for bills charts    ######################################

#PIE CHART DATA MANIPULATION FOR BILLS 
# Home > Prediction Input Page > Survival Prediction tab

gross = bills2.groupby(['Consolidated.Main.Group']).sum()
gross_list = {}
for row in gross.itertuples():
    gross_list[row.Index] = row[1]

category_list = list(gross_list.keys())

#BAR CHART DATA MANIPULATION FOR BILLS
# Home > Prediction Input Page > Survival Prediction tab

average = bills2.groupby(['Consolidated.Main.Group']).mean()
average_list = {}
display_values = [] #text on bars
key_values = [] # category 
all_values = [] # category average expenditure values
for row in average.itertuples():
    average_list[row.Index] = row[1]

print('###########################################', average)
print('###############################BEFORE############', average_list)

average_list = sorted(average_list.items(), key=operator.itemgetter(1), reverse=True)
print('###################################AFTER########avg list:', average_list)

for i in average_list:
    key_values.append(i[0])
    all_values.append(round(i[1],2))

for value in all_values:
    display_values.append('$ ' + str(value))



###########################################   Data manipulation for clinical charts    ######################################

# Function to rename dictionary's keys
def rename_keys(dict_, new_keys):
    """
     new_keys: type List(), must match length of dict_
    """
    d1 = dict( zip( list(dict_.keys()), new_keys) )
    return {d1[oldK]: value for oldK, value in dict_.items()}

# Calculate percentage of clinical data value
def calPercent(df1,columnDF,nullExist=False, *replaceWith):
    dict_list = {}
    values = columnDF.unique()
    if nullExist:
        values = np.insert(values,0,replaceWith)
        checker = False
        for i in range(len(values)):
            if isinstance(values[i],float):
                checker = True
        if checker:
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


# Transform data columns into dataframe
def df_func(output,column1,column2):
    tmp = pd.DataFrame(output.items(), columns=[column1, column2])
    return tmp

# Data manipulation for ER/ PR charts
# Home > Dashboard > Clinical
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

#Execution of functions for clinical

# # Remove null values
death_cause = clinical['cause_of_death']
death_cause_dict_old = calPercent(clinical,death_cause,True,"Alive")
death_cause_dict = rename_keys(death_cause_dict_old,\
                       death_cause.unique())

# (Hover) Binning of diagnosed age
bins = np.arange(clinical['Age_@_Dx'].min(),clinical['Age_@_Dx'].max() + 4, 4)
clinical['binned'] = np.searchsorted(bins, clinical['Age_@_Dx'].values)
age_bin_count = clinical.groupby(pd.cut(clinical['Age_@_Dx'], bins=19, precision = 0, right = False)).size()

er_finalized_dict = generate_epr_chart_data(clinical)

#Replace values from Column Race containing unusual values
clinical['Race'].replace('9', 'Unknown', inplace=True)
clinical['Race'].replace('#N/A', 'Unknown',inplace=True)
clinical['Race'].fillna('Unknown', inplace=True)
Race_List = clinical['Race'].unique()
Race_List = list(Race_List) + ['All'] #Keep because makes sense to display All

generate_waffle_chart()


#################################################### Functions for dashboard filter panel ##################################################
# Purpose: manipulate data according to filter inputs
# Home > Dashboard

def filter_df_all(df, min1, max1, tnm_select, er_select, pr_select, her2_select, race_select):
    '''
        df:dataframe
        age_slider: slider values in a tuple (min, max) - use age_slider[0 or 1] to extract
        er_status: str
        pr_status: str
        her2_status: str
        race_selec: str
        Purpose of this function filters the full dataset, clinical, using the variables given in the filter panel of the application.

    '''
    #checking for parameter == 'All' in params
    dict_tmp = {
                    'ER': er_select,
                    'PR': pr_select,
                    'Stage': tnm_select,
                    'Her2': her2_select,
                    'Race': race_select
                }
    condition = (df['Age_@_Dx'] <= max1) & (df['Age_@_Dx'] >= min1) 
    output = df[condition]
    for k,v in dict_tmp.items():
        if v != "All":
            output = output[(output[k] == v)]

    return output

def filter_df_wo_stage(df, min1, max1, er_select, pr_select, her2_select, race_select):
    '''
        Same functionality as filter_df_all without tnm stage
        This is for a specific graph - TNM Stage Alive VS Dead
    '''
    dict_tmp = {
                    'ER': er_select,
                    'PR': pr_select,
                    'Her2': her2_select,
                    'Race': race_select
    }
    condition = (df['Age_@_Dx'] <= max1) & (df['Age_@_Dx'] >= min1) 
    output = df[condition]
    for k,v in dict_tmp.items():
        if v != "All":
            output = output[(output[k] == v)]
    return output

def filter_df_epr_chart(df, min1, max1, tnm_select):
    '''
        Same function as filter_df_all as well.
        This is for the relationship between er & pr chart.
    
    '''
    dict_tmp = {'Stage': tnm_select}
    condition = (df['Age_@_Dx'] <= max1) & (df['Age_@_Dx'] >= min1) 
    output = df[condition]
    for k,v in dict_tmp.items():
        if v != "All":
            output = output[(output[k] == v)]
    return output


# Display scatter plot for transaction spending of each patient ID

def cost_scatter_data(min_n, max_n):
    '''
        Total Gross Cost Dataset Generator.
        For selected number of patients, show scatter plot
    ''' 
    scatter = bills2.groupby(['Case.No']).sum()
    scatter = scatter.reset_index()
    # print(scatter)
    new_scatter = scatter[min_n:max_n]
    
    scatter_list = {} #convert new_scatter to dictionary
    for row in new_scatter.itertuples():
        scatter_list[row.Index] = row[1]

    # pd.DataFrame.from_dict(scatter_list)

    return new_scatter

# Radio buttons for filter panel
def display_radio_btns():
    '''
        returns radio buttons selector to show TNM filter for Kaplan
    '''
    return dcc.RadioItems(
    id='radio-list',
    options=[
        {'label': 'Kaplan Meier', 'value': 'Kaplan Meier'},
        {'label': 'Clinical', 'value': 'Clinical'},
    ],
        value='Clinical',
        inputStyle={"margin-right": "5px", 'margin-left':'10px'}
    )  

# Filter panel design for clinical tab
# Home > Dashboard > Clinical

def generate_clinical_controls():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card-clinical",
        children=[
            display_radio_btns(),
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
            html.P("Filter By Race:"),
            dcc.Dropdown(
                id="race_select",
                options=[{"label":i, "value":i} for i in Race_List],
                value=Race_List[0]
            ),
            html.Div([
                html.Br(),
                html.Div(id='hide-this',children=[html.P("Select TNM Stage")]),
                dcc.Dropdown(
                    id="tnm_select",
                    options= [{"label": i, "value": i} for i in tnm_list],
                    value=tnm_list[1]
                ),
            ],style={'display':'block'}),
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
            html.Div(id="hide-me-1",
            children=[
                    html.Br(),
                    html.P("Select T Stage:"),
                    dcc.Dropdown(
                        id="t_select",
                        value=['t0'], #default values
                    )
                ],style={'display': 'none'}
            ),
            html.Div(id="hide-me-2", 
            children=[
                    html.Br(),
                    html.P("Select N Stage:"),
                    dcc.Dropdown(
                        id="n_select",
                        value=['n0'], #default values
                        
                    )
                ],style={'display':'none'}
            ),
            html.Div(id="hide-me-3",children=[
                html.Br(),
                html.P("Select M Stage:"),
                dcc.Dropdown(
                    id="m_select",
                    value=['m0'], #default values
                ),
            ],style={'display':'none'}),
            html.Br(),
            html.Br(),
            html.Br(),
        ],
    )

# Filter panel design for bills tab
# Home > Dashboard > Bills

def generate_bills_controls():
    """
    :return: A Div containing controls for graphs. (Bills)
    """
    return html.Div(
        id="control-card-bills",
        children=[

            html.Br(),
            html.P("Select Category"),
            dcc.Checklist(
                id="category_check",
                options= [{"label": i, "value": i} for i in category_list],
                value=category_list,
                labelStyle={'display': 'block'}
            ),
            html.Br(),
            html.P('Select Sample Size For Scatter Plot:'),
            dcc.RangeSlider(
                            id="cost_slider",
                            min=0,
                            max=10000,
                            value=[3000, 6500],
                            marks={
                                0: '0',
                                2000: '2000',
                                4000: '4000',
                                6000: '6000',
                                8000: '8000'
                            }
            ),

            html.Br(),
            html.Br()
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


################################################### Page layout design for bills tab #######################################################
# Home > Dashboard > Bills

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

################################################### Page layout design for clinical tab #######################################################
# Home > Dashboard > Clinical

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

################################################### Chart dashboard design for bills tab #######################################################
# Home > Dashboard > Bills

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
                                        generate_bills_controls(),
                                        
                                    ], className="bill_cardbody_layout"
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
                                                            html.H2("Bills Dataset"),
                                                        ),
                                                        dbc.CardBody(
                                                            [
                                                                dbc.Container(
                                                                    [
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
                                                                                                        x= [],
                                                                                                        y=[],
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
                                                                                                xaxis = {'title': 'Patient ID','automargin': True},
                                                                                                yaxis = {'title': 'Patient expenditure ($)','automargin': True}
                                                                                            )
                                                                                            },                
                                                                                    )
                                                                                
                                                                            ),
                                                                            dbc.Col(
                                                                                dcc.Graph(
                                                                                    id='svc-pie-chart',
                                                                                    figure={
                                                                                        'data': [
                                                                                            go.Pie(
                                                                                                labels = list(gross_list.keys()),
                                                                                                values = list(gross_list.values())
                                                                                            )
                                                                                        ],
                                                                                        'layout': go.Layout(
                                                                                            title='Gross Expenditure by Category',
                                                                                            hovermode='closest'
                                                                                        ),
                                                                                        
                                                                                    }
                                                                                )
                                                                            )
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                dcc.Graph(
                                                                                    id='average_service',
                                                                                    figure={
                                                                                        'data': [
                                                                                            go.Bar(
                                                                                                x= key_values,
                                                                                                y= all_values,
                                                                                                text=display_values,
                                                                                                textposition='auto',
                                                                                                marker={'color': ['#1881e1','#1574ca', '#1367b4','#105a9d','#0e4d87','#0c4070','#09335a', '#072643','#04192d','#020c16', '#000000']}
                                                                                            )
                                                                                        ],
                                                                                        'layout': go.Layout(
                                                                                            title='Average Expenditure by Category',
                                                                                            xaxis = {'title':'Category'},
                                                                                            yaxis={'title':'Average Patient Expenditure ($)'},
                                                                                            hovermode='closest'
                                                                                        ),
                                                                                        
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
                    ), width = {"size":9},
                )
            ]
        )
    ]
)

################################################### Chart dashboard design for clinical tab #######################################################
# Home > Dashboard > Clinical

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
                                        #html.Div([display_radio_btns()], style={'display': "inline-block"}),
                                        html.Br(),
                                        generate_clinical_controls()

                                    ], className="clinical_cardbody_layout"
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
                                                                                                xaxis = {'title': 'Cause of Death', 'automargin': True},
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
                                                                        ),
                                                                        html.Div(id="hide-km-charts", children=[
                                                                            dbc.Row(
                                                                                [
                                                                                    dbc.Col(
                                                                                        [
                                                                                        dcc.Graph(
                                                                                            id="OS_KM",
                                                                                            figure=go.Figure(
                                                                                                data=[],
                                                                                                layout=go.Layout(
                                                                                                    title="Patient's Overall Survival Kaplan Meier Chart",
                                                                                                    xaxis = {'title': 'Year'},
                                                                                                    yaxis = {'title': 'Percentage of Survival'},
                                                                                                    hovermode= "closest",
                                                                                                ),
                                                                                            ),
                                                                                        ),
                                                                                        ], 
                                                                                    ),  
                                                                                ], 
                                                                            ),
                                                                            dbc.Row(
                                                                                [
                                                                                    dbc.Col(
                                                                                        [
                                                                                        dcc.Graph(
                                                                                            id="CSS_KM",
                                                                                            figure=go.Figure(
                                                                                                data=[],
                                                                                                layout=go.Layout(
                                                                                                    title="Patient's Cancer Specific Survival Kaplan Meier Chart",
                                                                                                    #height=400,
                                                                                                    #width=1200,
                                                                                                    xaxis = {'title': 'Year'},
                                                                                                    yaxis = {'title': 'Percentage of Survival'}, 
                                                                                                    hovermode= "closest",
                                                                                                ),
                                                                                            ),
                                                                                        ),
                                                                                        ],
                                                                                    ), 
                                                                                ], 
                                                                            ),
                                                                            dbc.Row(
                                                                                [
                                                                                    dbc.Col(
                                                                                        [
                                                                                        dcc.Graph(
                                                                                            id="DFS_KM",
                                                                                            figure=go.Figure(
                                                                                                data=[],
                                                                                                layout=go.Layout(
                                                                                                    title="Patient's Disease-Free Survival Kaplan Meier Chart",
                                                                                                    xaxis = {'title': 'Year'},
                                                                                                    yaxis = {'title': 'Percentage of Survival'},                                                                
                                                                                                    hovermode= "closest",
                                                                                                ),
                                                                                            ),
                                                                                        )
                                                                                        ],
                                                                                    ), 
                                                                                ], 
                                                                            ),
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
                                        
################################################## Design for prediction page #######################################################                                                             

# HTML for prediction tab
# Home > Prediction Input Page > Survival Prediction tab
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
                [
                    html.H1("Survival Prediction")
                ], align="center", justify="center",
            ),   
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

cost_layout = dbc.Container(
        [dbc.Row(
            #dbc.Col(
            [
                html.H1("Cost Prediction")
            ], align="center", justify="center",
        ),  


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

doctor_items = [
    dbc.DropdownMenuItem("Doctor View", href="/survival/", active=True),
    dbc.DropdownMenuItem("Patient View", href="/survival/patient"),
]

patient_items = [
    dbc.DropdownMenuItem("Doctor View", href="/survival/"),
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

###################################### Code to create a Dash app (server configuration) ###################################
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
                         external_scripts=external_scripts
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
    ])

    #Initialize callbacks after our app is loaded
    # Pass dash_app as a parameter
    init_callbacks(dash_app)

    return dash_app.server


########################### Callback for dash app ################################################
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

        elif pathname == "/survival/":
            cookies = session['received']
            cookies = cookies.decode("utf-8")

            patient = pd.read_csv("..\\middleWomen\\patient_new.csv")

            x = patient["x"]
            y = patient["y"]

            surv4 = (go.Scatter(x=x/365.25, y=y*100, name="hv",
                                line_shape='hv'))

            #doctor's graphs
            # Home > Prediction Input Page > Survival Prediction tab > Doctor view
            doctor_graphs =  html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                html.H1("Patient's Predicted Kaplan Meier Chart")
                                            ),
                                            dbc.CardBody(
                                                [
                                                    dcc.Graph(
                                                        id="Kaplan Meier",
                                                        figure=go.Figure(
                                                            data=[surv4],
                                                            layout=go.Layout(
                                                                height=650,
                                                                xaxis_range=(0, 10),
                                                                yaxis_range=(0, 100),
                                                                xaxis = {'title': 'Year(s)'},
                                                                yaxis = {'title': 'Percentage of Survival (%)'}, 
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                                    ),                                                
                                     
                                                ]
                                            )
                                        ]
                                    )
                                ), width={"size":10, "offset":1}
                            )
                        ]
  
                    ),                
                ]
            )
            return survival_layout, doctor_button, doctor_graphs
        elif pathname == "/cost/":
  
            #cost outputs

            my_bills = pd.read_csv("..\\middleWomen\\bills_new.csv")
            my_bills = my_bills.iloc[:,:-1]
            # key = ["6 months after","1 year after","2 year after","5 years after","10 years after"]
            key = my_bills.columns.tolist()[1:]
            values = [my_bills[k].tolist() for k in key]

            #Prepare chart format for cost data - Financial Consultant
            trace1 = go.Bar(
                                x=key,
                                y=[round(x[0],2) for x in values],
                                text=[round(x[0],2) for x in values],
                                textposition='auto',
                                name = "predicted cost"
            )

            trace2= go.Scatter(
            x=key,
            y=[round(x[0],2) for x in values],
            name = "prediction line"
            )

            # Home > Prediction Input Page > Cost Prediction tab
            cost_graphs =  html.Div(
                [
                    dbc.Container(
                        [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        dbc.Card(
                                            [
                                                dbc.CardHeader(
                                                    html.H2("Predicted Cost Table")
                                                ),
                                                dbc.CardBody(
                                                    [
                                                        dcc.Graph(
                                                            id='Cost Table',
                                                            figure={
                                                            'data': [
                                                                go.Table(
                                                                    header=dict(values=["<b>Year(s)</b>","<b>Cost($)</b>"],
                                                                    fill_color='paleturquoise',
                                                                    align='center'),
                                                                    cells=dict(
                                                                        values=[key, [[round(x[0],2)] for x in values]],
                                                                        fill_color='white',
                                                                        align='center'
                                                                    )
                                                                )
                                                            ],
                                                            'layout':go.Layout(

                                                            )
                                                        })                                                        
                                                    ]
                                                )
                                            ]
                                        )
                                    ),width = {"size":5, "offset": 1}           
                                ),
                                dbc.Col(
                                    html.Div(
                                        dbc.Card(
                                            [
                                                dbc.CardHeader(
                                                    html.H2("Patient's Cost Prediction ($)")
                                                ),
                                                dcc.Graph(
                                                    id="Cost Prediction",
                                                    figure=go.Figure(
                                                        data=[trace1,trace2],
                                                        layout=go.Layout(
                                                            xaxis = {'title': 'Year(s)'},
                                                            yaxis = {'title': 'Fees Prediction ($)'},                                                             


                                                    
                                                        )
                                                    ),
                                                )
                                            ]
                                        )
                                    ), width={"size":5}
                                )
                            ]
                        ),
                        ]

                    )
                ]

            )

        
            return cost_layout, cost_graphs
        elif pathname =="/survival/patient":

            #survival outputs

            surv = pd.read_csv("..\\middleWomen\\survival.csv")
            # key = ["6 months after","1 year after","2 year after","5 years after","10 years after"]
            surv_key = surv.columns.tolist()[1:]
            surv_values = [surv[k].tolist() for k in surv_key]


            #Prepare chart format for patient survival data

            trace1s = go.Bar(
                x=surv_key,
                y=[round(n[0]*100,2) for n in surv_values],
                text=[round(n[0]*100,2) for n in surv_values],
                textposition='auto',
                name = "survival rate"
            )

            trace2s= go.Scatter(
                x=surv_key,
                y=[round(n[0]*100,2) for n in surv_values],
                name = "survival trendline"
                )


            #patient's graphs
            # Home > Prediction Input Page > Patient Prediction tab
            patient_graphs =  html.Div(
                [
                    dbc.Container(
                        [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        dbc.Card(
                                            [
                                                dbc.CardHeader(
                                                    html.H2("Survival Rate Table")
                                                ),
                                                dbc.CardBody(
                                                    [
                                                        dcc.Graph(
                                                            id='Survival Rate Table',
                                                            figure={
                                                                'data': [
                                                                    go.Table(
                                                                        header=dict(
                                                                                values=["<b>Year(s)</b>","<b>Survival Rate (%)</b>"],
                                                                        fill_color='paleturquoise',
                                                                        align='center'),
                                                                        cells=dict(
                                                                            values=[surv_key, [[round(x[0]*100,2)] for x in surv_values]],
                                                                            fill_color='white',
                                                                            align='center')
                                                                    )],
                                                                    'layout': go.Layout(

                                                                    )
                                                                }
                                                        ),                                                        
                                                    ]
                                                )
                                            ]
                                        )
                                    ),width = {"size":5, "offset": 1}           
                                ),
                                dbc.Col(
                                    html.Div(
                                        dbc.Card(
                                            [
                                                dbc.CardHeader(
                                                    html.H2("Survival Rate Chart")
                                                ),
                                                dbc.CardBody(
                                                    dcc.Graph(
                                                        id="Survival Prediction",
                                                        figure=go.Figure(
                                                            data=[trace1s,trace2s],
                                                            layout=go.Layout(

                                                            xaxis = {'title': 'Year(s)'},
                                                            yaxis = {'title': 'Percentage of Survival (%)'}, 
                                                            )
                                                        ),
                                                    )
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
                                                    html.H2("Breast Cancer Patients Survival Rate")
                                                ),
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            [
                                                                    html.Img(
                                                                    src=dash_app.get_asset_url('waffle-chart-survived.png'),
                                                                    id="waffle-2",
                                                                    style={
                                                                    "height": "300px",
                                                                    "width": "95%",
                                                                    "margin-bottom": "3%",
                                                                    "margin-left":"3%",
                                                                    "margin-right":"3%",
                                                                    "margin-top":"3%"
                                                                
                                                                    },
                                                                    ),
                                                                html.P(
                                                                    "Out of 100 random breast cancer patients, X will survive within the 10 year time period.",
                                                                    style={"margin-left":"3%", "margin-bottom":"3%"}
                                                                )
                                                            ]
                                                        )
                                                
                                                    ]                                              
                                                )
                                            ]
                                        )
                            
                                    ), width ={"size":10, "offset":1}
                                ),
  
                            ]
                        )

                        ]

                    )
                ]

            )
 
            return survival_layout, patient_button, patient_graphs

    ##################################Interactive Dynamic Dropdown for dashboard filter (backend)#############################################
    @dash_app.callback(
        dash.dependencies.Output('t_select', 'options'),
        [
            dash.dependencies.Input('age_slider', 'value'),
            dash.dependencies.Input('er_select','value'),
            dash.dependencies.Input('pr_select','value'),
            dash.dependencies.Input('her2_select','value'),
            dash.dependencies.Input('race_select','value')
        ]
    )
    def update_t_dropdown(age_slider, er_select, pr_select, her2_select, race_select):
        #This Is Filtered according to available values in dataset
        #filter full dataset by selected val, get available values in N column
        #Assuming T, N, M stages that were entered were all vald and according to ajcc staging
        max1=age_slider[1]
        min1 = age_slider[0]

        condition = (input_df['Age_@_Dx'] <= max1) & (input_df['Age_@_Dx'] >= min1)
        output = input_df[condition]

        dict_tmp2 = {
            'ER': er_select,
            'PR': pr_select,
            'Her2': her2_select,
            'Race': race_select,
        }

        for k,v in dict_tmp2.items():
    
            if v != "All":
                output = output[(output[k] == v)]

        valid = list(output['T'].unique()) 
        return [{'label': i, 'value': i} for i in sorted(valid)] 

    #Dynamic Dropdown
    @dash_app.callback(
        dash.dependencies.Output('n_select', 'options'),
        [
            dash.dependencies.Input('age_slider', 'value'),
            dash.dependencies.Input('er_select','value'),
            dash.dependencies.Input('pr_select','value'),
            dash.dependencies.Input('her2_select','value'),
            dash.dependencies.Input('race_select','value'),
            dash.dependencies.Input('t_select', 'value')
        ]
    )
    def update_n_dropdown(age_slider, er_select, pr_select, her2_select, race_select,t_value):
        #This Is Filtered according to available values in dataset
        #filter full dataset by selected val, get available values in N column
        #Assuming T, N, M stages that were entered were all vald and according to ajcc staging
        max1=age_slider[1]
        min1 = age_slider[0]

        condition = (input_df['Age_@_Dx'] <= max1) & (input_df['Age_@_Dx'] >= min1)
        output = input_df[condition]

        dict_tmp2 = {
            'ER': er_select,
            'PR': pr_select,
            'Her2': her2_select,
            'Race': race_select,
            'T': t_value
        }

        for k,v in dict_tmp2.items():
            if v != "All":
                output = output[(output[k] == v)]
        valid = list(output['N'].unique()) #Get valid N based on  T values entered
        return [{'label': i, 'value': i} for i in sorted(valid)] #populate N dropdown with new n values

    @dash_app.callback(
        dash.dependencies.Output('m_select', 'options'),
        [
            dash.dependencies.Input('age_slider', 'value'),
            dash.dependencies.Input('er_select','value'),
            dash.dependencies.Input('pr_select','value'),
            dash.dependencies.Input('her2_select','value'),
            dash.dependencies.Input('race_select','value'),
            dash.dependencies.Input('t_select', 'value'),
            dash.dependencies.Input('n_select', 'value'),
        ])
    def update_m_dropdown(age_slider, er_select, pr_select, her2_select, race_select,t_value, n_value ):
        #optimise code later - delete repeated code
        max1=age_slider[1]
        min1 = age_slider[0]

        condition = (input_df['Age_@_Dx'] <= max1) & (input_df['Age_@_Dx'] >= min1)
        output = input_df[condition]
        print(input_df.head())
        dict_tmp = {
            'ER': er_select,
            'PR': pr_select,
            'Her2': her2_select,
            'Race': race_select,
            'T': t_value,
            'N': n_value
        }

        for k,v in dict_tmp.items():
            if v != "All":
                output = output[(output[k] == v)]

        valid = list(output['M'].unique())
        return [{'label': i, 'value': i} for i in sorted(valid)]
    
    ### Hide & Show Dropdowns According to radio buttons
    
    @dash_app.callback(
        [
            dash.dependencies.Output('hide-me-1','style'), 
            dash.dependencies.Output('t_select','style'), 
            dash.dependencies.Output('hide-km-charts', 'style')
        ],
        [dash.dependencies.Input('radio-list','value')]
    )
    def toggle_dd_visibility(visible):
        if visible == 'Kaplan Meier':
            return {'display':'block'}, {'display':'block'}, {'display':'block'}
        if visible == 'Clinical':
            return {'display':'none'},{'display':'none'}, {'display':'none'}
    
    @dash_app.callback(
        [
            dash.dependencies.Output('hide-me-2','style'), 
            dash.dependencies.Output('n_select','style')
        ],
        [dash.dependencies.Input('radio-list','value')]
    )
    def toggle_d2_visibility(vi):
        if vi == 'Kaplan Meier':
            return {'display':'block'}, {'display':'block'}
        if vi == 'Clinical':
            return {'display':'none'},{'display':'none'}

    @dash_app.callback(
        [
            dash.dependencies.Output('hide-me-3','style'), 
            dash.dependencies.Output('m_select','style')
        ],
        [dash.dependencies.Input('radio-list','value')]
    )
    def toggle_d3_visibility(vis):
        if vis == 'Kaplan Meier':
            return {'display':'block'}, {'display':'block'}
        if vis == 'Clinical':
            return {'display':'none'}, {'display':'none'}
    

    @dash_app.callback(
        [
            dash.dependencies.Output('hide-this','style'), 
            dash.dependencies.Output('tnm_select','style')
        ]
        ,
        [dash.dependencies.Input('radio-list','value')]
    )
    def toggle_dropdown(v):
        if v == 'Kaplan Meier':
            return {'display':'none'}, {'display':'none'}
        if v == 'Clinical':
            return {'display':'block'}, {'display':'block'}

    @dash_app.callback(
        [
            dash.dependencies.Output('age-distribution-hist', 'figure'),
            dash.dependencies.Output('alive_dead_bar','figure'),
            dash.dependencies.Output('er_pr_chart','figure'),
            dash.dependencies.Output('OS_KM','figure'),
            dash.dependencies.Output('CSS_KM','figure'),
            dash.dependencies.Output('DFS_KM','figure')
        ],
        [
            dash.dependencies.Input('age_slider', 'value'),
            dash.dependencies.Input('tnm_select','value'),
            dash.dependencies.Input('er_select','value'),
            dash.dependencies.Input('pr_select','value'),
            dash.dependencies.Input('her2_select','value'),
            dash.dependencies.Input('race_select','value'),
            dash.dependencies.Input('t_select','value'),
            dash.dependencies.Input('n_select','value'),
            dash.dependencies.Input('m_select','value')
        ]
    )

    def update_clinical(age_slider,tnm_select, er_select, pr_select, her2_select, race_select,t_select,n_select,m_select ):

        filters_dict = {
                    "age_lower": age_slider[0],
                    "age_upper":age_slider[1],
                    'Race': race_select,
                    'T': t_select,
                    'N': n_select,
                    'M': m_select,
                    "ER": er_select,
                    "PR": pr_select,
                    'Her2':her2_select
                }

        #Slice Df according to inputs in filter
        df = filter_df_all(clinical, age_slider[0], age_slider[1] , tnm_select, er_select, pr_select, her2_select,race_select)

        #editing alive vs dead bar chart - overwrite initial version
        cause = df['cause_of_death']
        dcdict_new = calPercent(df,cause,True,"Alive")

        dcdict = rename_keys(dcdict_new,\
                                ['Alive', 'Dead- Breast Cancer', 'Dead- Others', 'Dead- Unknown'])

        #Overwrite original chart data with sliced dataset according to filters
        epr_df = filter_df_epr_chart(clinical, age_slider[0], age_slider[1] , tnm_select)
        er_dict = generate_epr_chart_data(epr_df)

        #overall survival Kaplan Meier chart
        km_os,km_dfs,km_css = KM2.generate_kaplan_meier_with_filters_for_all_survival_types(filters_dict,input_df) #all filters wo tnm stage

        
        if km_os.shape[0] == 0:
            x = [0]
            y = [0]
            lower = [0]
            upper = [0]
        else:
            x = km_os["time"].tolist()
            y = km_os["estimate"].tolist()
            lower = km_os["lower"].tolist()
            upper = km_os["upper"].tolist()
        
        km_upper = go.Scatter(x=x, y=y*100,
            fill=None,
            mode='lines',
            line_color='indigo',
            name='Overall Survival',
        )

        km_lower = go.Scatter( x=x, y=upper*100,
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines', 
            line_color='lightblue',
            name="{}".format('95% Upper CI'),
        )
        km = go.Scatter( x=x, y=lower*100,
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines', 
            line_color='lightblue',
            name="{}".format("95% Lower CI"),
        )
        os_new = [km_upper,km_lower,km]

        #dfs Kaplan Meier chart 
        if km_dfs.shape[0] == 0:
            dfs_x = [0]
            dfs_y = [0]
            dfs_lower = [0]
            dfs_upper = [0]
        else:
            dfs_x = km_dfs["time"].tolist()
            dfs_y = km_dfs["estimate"].tolist()
            dfs_lower = km_dfs["lower"].tolist()
            dfs_upper = km_dfs["upper"].tolist()

        dfs_km_upper = go.Scatter(x=dfs_x, y=dfs_y*100,
            fill=None,
            mode='lines',
            line_color='indigo',
            name='Disease Free Survival',
        )

        dfs_km_lower = go.Scatter(x=dfs_x,
            y=dfs_upper*100,
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines', 
            line_color='lightblue',
            name="{}".format('95% Upper CI'),
        )
        dfs_km = go.Scatter(x=dfs_x,
            y=dfs_lower*100,
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines', 
            line_color='lightblue',
            name="{}".format("95% Lower CI"),
        )
        dfs_new = [dfs_km_upper,dfs_km_lower,dfs_km]
        #css Kaplan Meier chart
        if km_css.shape[0] == 0:
            css_x = [0]
            css_y = [0]
            css_lower = [0]
            css_upper = [0]
        else:
            css_x = km_css["time"].tolist()
            css_y = km_css["estimate"].tolist()
            css_lower = km_css["lower"].tolist()
            css_upper = km_css["upper"].tolist()

        css_km_upper = go.Scatter(x=css_x, y=css_y*100,
            fill=None,
            mode='lines',
            line_color='indigo',
            name='Cancer Specific Survival',
        )

        css_km_lower = go.Scatter( x=css_x,
            y=css_upper*100,
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines', 
            line_color='lightblue',
            name="{}".format('95% Upper CI'),
        )
        css_km = go.Scatter( x=css_x,
            y=css_lower*100,
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines', 
            line_color='lightblue',
            name="{}".format("95% Lower CI"),
        )
        css_new = [css_km_upper,css_km_lower,css_km]
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
            ],
            'layout': go.Layout(
                        title = "Patient's Diagnosed Age Distribution",
                        xaxis = {'title': 'Diagnosed Age'},
                        yaxis = {'title': 'Percentage of Patients'},
                    )
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
            ],
            'layout': go.Layout(
                    title = "Proportion of Patients Alive Vs Dead",
                    xaxis = {'title': 'Cause of Death','automargin': True},
                    yaxis = {'title': 'Percentage of Patients'},
                    hovermode='closest'
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
                            hovermode='event+closest',
                            barmode='stack'
                        )
                    }

        figure5 = {
                'data':os_new,
                'layout':go.Layout(
                        title="Patient's Overall Survival Kaplan Meier Chart",
                        height=400,
                        width=1200,  
                        xaxis = {'title': 'Year'},
                        yaxis = {'title': 'Percentage of Survival'}, 
                        hovermode= "closest",
                    )
            }

        figure6 = {
                'data': dfs_new,
                'layout':go.Layout(
                        title="Patient's Disease-Free Survival Kaplan Meier Chart",
                        height=400,
                        width=1200,
                        xaxis = {'title': 'Year'},
                        yaxis = {'title': 'Percentage of Survival'}, 
                        hovermode= "closest",
                    )

            }

        figure7 = {
                'data': css_new,
                'layout':go.Layout(
                    title="Patient's Cancer Specific Survival Kaplan Meier Chart",
                    height=400,
                    width=1200,
                    xaxis = {'title': 'Year'},
                    yaxis = {'title': 'Percentage of Survival'}, 
                    hovermode= "closest",
                )
            }

            
        return figure, figure2, figure4, figure5, figure6, figure7


    @dash_app.callback(
            dash.dependencies.Output('expenditure-scatter', 'figure'),
            [dash.dependencies.Input('cost_slider', 'value')]
            )
    def update_cost(cost_slider):
        data = []
        # print(cost_slider)
        min_patient_num = cost_slider[0]
        max_patient_num = cost_slider[1]
        patient_spend = cost_scatter_data(min_patient_num, max_patient_num)
        patient_spend[cost_slider[0]: cost_slider[1]]
        patient_id = list(patient_spend['Case.No'])
        total_spent = list(patient_spend['Gross..exclude.GST.'])
        data.append(
            go.Scatter(
                x=patient_id,
                y=total_spent,
                mode='markers',
                marker=dict(
                size=12,
                color=np.random.randn(100000), #set color equal to a variable
                colorscale='Viridis', # one of plotly colorscales
                showscale=True
            )
            )
        )
        figure={
                            'data': data,
                            'layout': go.Layout(
                                title = "Distribution of patient expenditure",
                                xaxis = {'title': 'Patient ID','automargin': True},
                                yaxis = {'title': 'Patient expenditure ($)'})
            }
        return figure

    @dash_app.callback(
        dash.dependencies.Output('svc-pie-chart', 'figure'),
   
        [dash.dependencies.Input('category_check', 'value')]
    )
    def update_pie_chart(category_check):
        print(category_check)
        new_vals_list = []

        for c in category_check:
            new_vals_list.append(gross_list[c])

        fig = {
            'data':[
                    go.Pie(
                        labels = category_check,
                        values = new_vals_list
                        #marker = dict(color = '#97B2DE')
                    )
            ],
            'layout': go.Layout(
                    title='Gross Expenditure by Category',
                    hovermode='closest'
                )
            }
        return fig