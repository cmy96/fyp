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
import matplotlib.pyplot as plt
from matplotlib import cm
from pywaffle import Waffle
import math
import os, shutil
from flask_caching import Cache
import datetime



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
# app.config.suppress_callback_exceptions = True

cache = Cache(app.server, config={
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'filesystem',
    'CACHE_DEFAULT_TIMEOUT':20,
    'CACHE_THRESHOLD':3,
    'CACHE_DIR': "C:\\Users\\Jesslyn\\Documents\\GitHub\\fyp\\Frontend\\Code\\MAINAPP\\application\\cache"
})
app.config.suppress_callback_exceptions = True

timeout = 20
app.layout = html.Div([
    html.Div(id='flask-cache-memoized-children'),
    dcc.RadioItems(
        id='flask-cache-memoized-dropdown',
        options=[
            {'label': 'Option {}'.format(i), 'value': 'Option {}'.format(i)}
            for i in range(1, 4)
        ],
        value='Option 1'
    ),
    html.Div('Results are cached for {} seconds'.format(timeout))
])


@app.callback(
    Output('flask-cache-memoized-children', 'children'),
    [Input('flask-cache-memoized-dropdown', 'value')])
@cache.memoize(timeout=timeout)  # in seconds
def render(value):
    return 'Selected "{}" at "{}"'.format(
        value, datetime.datetime.now().strftime('%H:%M:%S')
    )



# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data
clinical = pd.read_csv(DATA_PATH.joinpath("clinical_full_data.csv"))
bills2 = pd.read_csv(DATA_PATH.joinpath("dropped.csv"))
bills = pd.read_csv(DATA_PATH.joinpath("bills.csv"))
# prices = pd.read_csv(DATA_PATH.joinpath("price_master.csv"))

#Set Variables
tnm_list = clinical["Stage"].dropna().unique()
ER_list = clinical['ER'].dropna().unique()
pr_list = clinical['PR'].dropna().unique()
Her2_list = clinical['Her2'].dropna().unique()
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

#################################### Get Data From Prediction Model ####################################
#survival model
s_output = {"6 months before":100.0, "6 months after":96.20, "1 year after":90.10, 
    "2 years after":86.90, "5 years after":80.09, "10 years after":71.22}
df2 = pd.DataFrame(s_output.items(), columns=['Years', 'Survival'])

#cost prediction model
c_output = {"6 months before":3882.80, "6 months after":13112.54, "1 year after":2230.19, 
    "2 years after":1736.58, "5 years after":11800.33, "10 years after":14917.57}
df = pd.DataFrame(c_output.items(), columns=['Years', 'Cost'])

###########################################   Prediction model visualisations    ######################################


def generate_waffle_chart():
    '''
        parameters: waffle2data should be the output of survival model
        return: waffle chart image in assets folder (for results page to extract)
    '''
 #Data should look like this ={'Disease-free Survival': 61, 'Overall Survival': 34, 'Dead in 10 years':5}
    km = pd.read_csv('data\\kaplan_meier_by_group.csv') #extract from km prediction model
    kmdict = {'DFS': 0, 'OS': 0, 'CSS':0} #use this as a template to input ur values
    
    #sum of all dfs, os, css values
    total_rows = km.shape[0]

    #fill kmdict with predicted values
    for key in kmdict:
        value = km[km['class_label'] == key].shape[0]
        kmdict[key] = int((value / total_rows)*100)

    kmdict = rename_keys(kmdict, ['Disease-Free Survival','Overall Survival', 'Dead in 10 years']) #rename keys in dict
    print(kmdict)

    plt.figure(
        FigureClass=Waffle, 
        rows=5, 
        values=kmdict, 
        colors=['#3CB371','#90EE90','#FF0000'],
        legend={
            'labels': ["{0} ({1})".format(k, v) for k, v in kmdict.items()],
            'loc': 'upper left', 'bbox_to_anchor': (1, 1)
            },
        icons='child', icon_size=14, 
        icon_legend=True,
        figsize=(10, 9),
        title={
        'label': 'Survivability Rate for Breast Cancer Patients ',
        'loc': 'center',
        'fontdict':{'fontsize':8}
        }
    )

    filename = 'C:\\Users\\Jesslyn\\Documents\\GitHub\\fyp\\Frontend\\Code\\MAINAPP\\application\\assets\\waffle-chart-km.png' #where picture will be stored, replace w ur own
    if os.path.exists(filename):
        os.remove(filename) #remove old
    plt.savefig(filename ,bbox_inches='tight', pad_inches=0) #replace 

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
            'label': 'Survivability for Breast Cancer Patients ',
            'loc': 'center',
            'fontdict':{'fontsize':8
            }}
    )
    filename = 'C:\\Users\\Jesslyn\\Documents\\GitHub\\fyp\\Frontend\\Code\\MAINAPP\\application\\assets\\waffle-chart-survived.png' #where picture will be stored
    if os.path.exists(filename):
        os.remove(filename) #remove old
    plt.savefig(filename ,bbox_inches='tight', pad_inches=0) #replace 



###########################################   Data manipulation for bills charts    ##########################r###########

#PIE CHART DE DONGXI
gross = bills2.groupby(['Consolidated.Main.Group']).sum()
gross_list = {}
for row in gross.itertuples():
    gross_list[row.Index] = row[1]

category_list = list(gross_list.keys())

#BAR CHART DE DONGXI
average = bills2.groupby(['Consolidated.Main.Group']).mean()
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

def df_func(output,column1,column2):
    tmp = pd.DataFrame(output.items(), columns=[column1, column2])
    return tmp
    
def generate_tnm_chart_data(df, dc,death_cause_dict_old):
    # for every stage in TNM_stage, it becomes the key for the MAIN dictionary
    #assuming no nan data
    TNM = df['Stage']
    # TNM_Stage = TNM.dropna().unique()
    TNM_dict={}

    for stage in clinical['Stage'].unique():
        status_dict = {}
        # tnm_death_cause_dict = calPercent(df,dc,True,"Alive")

        #every stage in Stage will have a dictionary that holds death_status as key and number of death as value
        for life_status in death_cause_dict_old.keys():
            tmp = df[['Stage','cause_of_death']]
            condition1 = df[TNM==stage]
            condition2 = tmp[(tmp['cause_of_death']==life_status) & (tmp['Stage']==stage)]
            condition3 = tmp[(tmp['cause_of_death'].isnull()) & (tmp['Stage'] == stage)]
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
        for stage in clinical['Stage'].unique():
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


#Execution of functions for clinical

# # this has null
death_cause = clinical['cause_of_death']
death_cause_dict_old = calPercent(clinical,death_cause,True,"Alive")
death_cause_dict = rename_keys(death_cause_dict_old,\
                       death_cause.unique())

# (Hover) Binning of diagnosed age - Tanny
bins = np.arange(clinical['Age_@_Dx'].min(),clinical['Age_@_Dx'].max() + 4, 4)
clinical['binned'] = np.searchsorted(bins, clinical['Age_@_Dx'].values)
age_bin_count = clinical.groupby(pd.cut(clinical['Age_@_Dx'], bins=19, precision = 0, right = False)).size()

finalized_dict = generate_tnm_chart_data(clinical, death_cause,death_cause_dict_old)
er_finalized_dict = generate_epr_chart_data(clinical)





###############################  This Section Is For Filters  ############################################################################################

def filter_df_all(df, min1, max1, tnm_select, er_select, pr_select, her2_select):
    '''
        df:dataframe
        age_slider: slider values in a tuple (min, max) - use age_slider[0 or 1] to extract
        er_status: str
        pr_status: str
        her2_status: str

        Purpose of this function filters the full dataset, clinical, using the variables given in the filter panel of the application.

    '''
    condition = (df['Age_@_Dx'] < max1) & (df['Age_@_Dx'] > min1) & (df['ER'] == er_select) & (df['PR'] == pr_select) & (df['Stage'] == tnm_select) & (df['Her2'] == her2_select)
    output = df[condition]
    return output

def filter_df_wo_stage(df, min1, max1, er_select, pr_select, her2_select):
    '''
        Same functionality as filter_df_all without tnm stage
        This is for a specific graph - TNM Stage Alive VS Dead
    '''
    condition = (df['Age_@_Dx'] <= max1) & (df['Age_@_Dx'] >= min1) & (df['ER'] == er_select) & (df['PR'] == pr_select) & (df['Her2'] == her2_select)
    output = df[condition]
    return output

def filter_df_epr_chart(df, min1, max1, tnm_select):
    '''
        Same function as filter_df_all as well.
        This is for the relationship between er & pr chart.
    
    '''
    condition = (df['Age_@_Dx'] < max1) & (df['Age_@_Dx'] > min1)  & (df['Stage'] == tnm_select) 
    output = df[condition]
    return output

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

def generate_clinical_controls():
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
            # html.Div(
            #     id="reset-btn-outer",
            #     children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            # ),
            html.Br()
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
                            value=[3000, 6500]
                            # marks={
                            #     0: '0',
                            #     2000: '2000',
                            #     4000: '4000',
                            #     6000: '6000',
                            #     8000: '8000'
                            # }
            ),

            html.Br(),
            # html.Div(
            #     id="reset-btn-outer",
            #     children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            # ),
            html.Br()
        ],
    )


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
                                                                                                xaxis = {'title': 'Patient ID'},
                                                                                                yaxis = {'title': 'Patient expenditure ($)'})
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


                                                                                                #marker = dict(color = '#97B2DE')
                                                                                            )
                                                                                        ],
                                                                                        'layout': go.Layout(
                                                                                            title='Gross Expenditure b Category',

                                                                                            #width = 500,
                                                                                            #height = 530,
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
                                        generate_clinical_controls()
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
                                                                                    # dcc.Graph(
                                                                                    #     id="tnm-stage-stacked-bar",
                                                                                    #     figure={
                                                                                    #         'data': [
                                                                                    #         go.Bar(
                                                                                    #                 x= finalized_dict['Alive'],
                                                                                    #                 y= clinical['Stage'].dropna().unique(),
                                                                                    #                 name='Alive',
                                                                                    #                 orientation='h',
                                                                                    #                 marker=dict(
                                                                                    #                 color='lightgreen',
                                                                                    #                 line=dict(color='lightgreen', width=3)
                                                                                    #                 )
                                                                                    #             ),
                                                                                    #             go.Bar(
                                                                                    #                 x= finalized_dict['breast cancer related'],
                                                                                    #                 y= clinical['Stage'].dropna().unique(),
                                                                                    #                 name='Dead- Breast cancer related',
                                                                                    #                 orientation='h',
                                                                                    #                 marker=dict(
                                                                                    #                 color='lightcoral',
                                                                                    #                 line=dict(color='lightcoral', width=3)
                                                                                    #                 )
                                                                                    #             ),
                                                                                    #             go.Bar(
                                                                                    #                 x=finalized_dict['n'],
                                                                                    #                 y= clinical['Stage'].dropna().unique(),
                                                                                    #                 name='Dead',
                                                                                    #                 orientation='h',
                                                                                    #                 marker=dict(
                                                                                    #                 color='indianred',
                                                                                    #                 line=dict(color='indianred', width=3)
                                                                                    #                 )
                                                                                    #             ),
                                                                                    #             go.Bar(
                                                                                    #                 x= finalized_dict['unknown'],
                                                                                    #                 y= clinical['Stage'].dropna().unique(),
                                                                                    #                 name='Unknown',
                                                                                    #                 orientation='h',
                                                                                    #                 marker=dict(
                                                                                    #                 color='lightslategrey',
                                                                                    #                 line=dict(color='lightslategrey', width=3)
                                                                                    #                 )
                                                                                    #             )
                                                                                    #         ],
                                                                                    #         'layout': go.Layout(
                                                                                    #             title = "TNM Stage Alive Vs Dead",
                                                                                    #             xaxis = {'title': 'Percentage of Patients'},
                                                                                    #             yaxis = {'title': 'Cancer Stages'},
                                                                                    #             hovermode='closest',
                                                                                    #             barmode='stack',
                                                                                    #             showlegend=False
                                                                                    #         )
                                                                                    #     }
                                                                                    # ),
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

cost_layout = dbc.Container(
        [dbc.Row(
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
            cookies = cookies.decode("utf-8")
            group =cookies.split(",")
            # # # cookie = html.H1(cookies)
            # print(str(group)[3:10], "group 1")
            km = pd.read_csv('data\\kaplan_meier_by_group.csv')
            g_os = km.loc[km['class_label'] == 'OS']
            os = g_os.loc[g_os['group_label'] == str(group)[3:10]]
            g_dfs = km.loc[km['class_label'] == 'DFS' ]
            dfs = g_dfs.loc[g_dfs['group_label'] == str(group)[3:10]]
            g_css = km.loc[km['class_label'] == 'CSS' ]
            css = g_css.loc[g_css['group_label'] == str(group)[3:10]]
            # #ok = json_normalize(cookies)
            # #ok = pd.read_json(cookies)
            # ok = pd.DataFrame(cookies)
            # print(ok)
            patient = pd.read_csv("..\\middleWomen\\patient_new.csv")
            #Execute for waffle
            generate_waffle_chart()

            x = patient["x"]
            y = patient["y"]

            surv4 = (go.Scatter(x=x/365.25, y=y*100, name="hv",
                                line_shape='hv'))

            #overall survival Kaplan Meier chart
            x = os["time"]
            y = os["estimate"]
            lower = os["lower"]
            upper = os["upper"]
            km_upper = go.Scatter(x=x, y=y*100,
                fill=None,
                mode='lines',
                line_color='indigo',
                name='Overall Survival',
            )

            km_lower = go.Scatter( x=x,
                y=upper*100,
                fill='tonexty', # fill area between trace0 and trace1
                mode='lines', 
                line_color='lightblue',
                name="{}".format('95% Upper CI'),
            )
            km = go.Scatter( x=x,
                y=lower*100,
                fill='tonexty', # fill area between trace0 and trace1
                mode='lines', 
                line_color='lightblue',
                name="{}".format("95% Lower CI"),
            )

            #dfs Kaplan Meier chart 
            dfs_x = dfs["time"]
            dfs_y = dfs["estimate"]
            dfs_lower = dfs["lower"]
            dfs_upper = dfs["upper"]
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

            #css Kaplan Meier chart
            css_x = css["time"]
            css_y = css["estimate"]
            css_lower = css["lower"]
            css_upper = css["upper"]
            css_km_upper = go.Scatter(x=css_x, y=css_y*100,
                fill=None,
                mode='lines',
                line_color='indigo',
                name='Cancer Specific Survival',
            )

            css_km_lower = go.Scatter( x=css_x,
                y=css_upper*100 ,
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

            #doctor's graphs
            doctor_graphs =  html.Div(
                [
                    html.Br(),
                    html.Br(),
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Survival Prediction",
                                        figure=go.Figure(
                                                            data=[km_upper,km_lower,km],
                                                            layout=go.Layout(
                                                                title="Patient's Overall Survival Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,  
                                                                xaxis_range=(0, 10),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'},
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                            dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Kaplan Meier",
                                        figure=go.Figure(
                                                            data=[dfs_km_upper,dfs_km_lower,dfs_km],
                                                            layout=go.Layout(
                                                                title="Patient's Disease-Free Survival Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,
                                                                xaxis_range=(0, 10),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'},                                                                
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Kaplan Meier",
                                        figure=go.Figure(
                                                            data=[css_km_upper,css_km_lower,css_km],
                                                            layout=go.Layout(
                                                                title="Patient's Cancer Specific Survival Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,
                                                                xaxis_range=(0, 10),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'}, 
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Kaplan Meier",
                                        figure=go.Figure(
                                                            data=[surv4],
                                                            layout=go.Layout(
                                                                title="Patient's Predicted Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,
                                                                xaxis_range=(0, 10),
                                                                yaxis_range=(0, 100),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'}, 
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                    
                ]
            )
            
            return survival_layout, doctor_button, doctor_graphs
            # return something
        elif pathname == "/survival/":
            cookies = session['received']
            cookies = cookies.decode("utf-8")
            group =cookies.split(",")
            # # # cookie = html.H1(cookies)
            # print(str(group)[3:10], "group 1")
            km = pd.read_csv('data\\kaplan_meier_by_group.csv')
            g_os = km.loc[km['class_label'] == 'OS']
            os = g_os.loc[g_os['group_label'] == str(group)[3:10]]
            g_dfs = km.loc[km['class_label'] == 'DFS' ]
            dfs = g_dfs.loc[g_dfs['group_label'] == str(group)[3:10]]
            g_css = km.loc[km['class_label'] == 'CSS' ]
            css = g_css.loc[g_css['group_label'] == str(group)[3:10]]
            # #ok = json_normalize(cookies)
            # #ok = pd.read_json(cookies)
            # ok = pd.DataFrame(cookies)
            # print(ok)
            patient = pd.read_csv("..\\middleWomen\\patient_new.csv")

            x = patient["x"]
            y = patient["y"]

            surv4 = (go.Scatter(x=x/365.25, y=y*100, name="hv",
                                line_shape='hv'))

            #overall survival Kaplan Meier chart
            x = os["time"]
            y = os["estimate"]
            lower = os["lower"]
            upper = os["upper"]
            km_upper = go.Scatter(x=x, y=y*100,
                fill=None,
                mode='lines',
                line_color='indigo',
                name='Overall Survival',
            )

            km_lower = go.Scatter( x=x,
                y=upper*100,
                fill='tonexty', # fill area between trace0 and trace1
                mode='lines', 
                line_color='lightblue',
                name="{}".format('95% Upper CI'),
            )
            km = go.Scatter( x=x,
                y=lower*100,
                fill='tonexty', # fill area between trace0 and trace1
                mode='lines', 
                line_color='lightblue',
                name="{}".format("95% Lower CI"),
            )

            #dfs Kaplan Meier chart 
            dfs_x = dfs["time"]
            dfs_y = dfs["estimate"]
            dfs_lower = dfs["lower"]
            dfs_upper = dfs["upper"]
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

            #css Kaplan Meier chart
            css_x = css["time"]
            css_y = css["estimate"]
            css_lower = css["lower"]
            css_upper = css["upper"]
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

            

            #doctor's graphs
            doctor_graphs =  html.Div(
                [
                    html.Br(),
                    html.Br(),
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Survival Prediction",
                                        figure=go.Figure(
                                                            data=[km_upper,km_lower,km],
                                                            layout=go.Layout(
                                                                title="Patient's Overal Survival Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,  
                                                                xaxis_range=(0, 10),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'}, 
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                            dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Kaplan Meier",
                                        figure=go.Figure(
                                                            data=[dfs_km_upper,dfs_km_lower,dfs_km],
                                                            layout=go.Layout(
                                                                title="Patient's Disease-Free Survival Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,
                                                                xaxis_range=(0, 10),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'}, 
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Kaplan Meier",
                                        figure=go.Figure(
                                                            data=[css_km_upper,css_km_lower,css_km],
                                                            layout=go.Layout(
                                                                title="Patient's Cancer Specific Survival Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,
                                                                xaxis_range=(0, 10),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'}, 
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Kaplan Meier",
                                        figure=go.Figure(
                                                            data=[surv4],
                                                            layout=go.Layout(
                                                                title="Patient's Predicted Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,
                                                                xaxis_range=(0, 10),
                                                                yaxis_range=(0, 100),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'}, 
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                    
                ]
            )
            return survival_layout, doctor_button, doctor_graphs
        elif pathname == "/cost/":
            #cost outputs
            my_bills = pd.read_csv("..\\middleWomen\\bills_new.csv")
            # key = ["6 months after","1 year after","2 year after","5 years after","10 years after"]
            key = my_bills.columns.tolist()[1:-1] #to be edited!!!!!!!!!!!!!!!!!!!!!1
            values = [my_bills[k].tolist() for k in key]

            #cost graphs for FC
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

            trace2= go.Scatter(
            x=key,
            y=values,
            name = "prediction line"
            )
            
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
                                                values=[key, [[round(x[0],2)] for x in values]],
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

        
            return cost_layout, cost_graphs
        elif pathname =="/survival/patient":

            #survival outputs
            surv = pd.read_csv("..\\middleWomen\\survival.csv")
            #Execute for waffle
            # generate_waffle_chart()
            # key = ["6 months after","1 year after","2 year after","5 years after","10 years after"]
            surv_key = surv.columns.tolist()[1:]
            surv_values = [surv[k].tolist() for k in surv_key]


            trace1s = go.Bar(
                x=surv_key,
                y=[round(n[0],2)*100 for n in surv_values],
                text=[round(n[0],2)*100 for n in surv_values],
                textposition='auto',
                name = "survival rate"
            )

            trace2s= go.Scatter(
                x=surv_key,
                y=[round(n[0],2)*100 for n in surv_values],
                name = "survival trendline"
            )


            #patient's graphs
            patient_graphs =  html.Div(
                [

                    dbc.Row(
                        dbc.Col(
                            html.Div(
                            [
                                dcc.Graph(
                                    id='Survivability Rate Table',
                                    figure={
                                    'data': [
                                    go.Table(
                                        header=dict(
                                                values=["<b>Years</b>","<b>Survivability Rate</b>"],
                                        fill_color='paleturquoise',
                                        align='center'),
                                        cells=dict(
                                            values=[surv_key, [[round(x[0],2)*100] for x in surv_values]],
                                            fill_color='white',
                                            align='center')
                                    )],
                                    'layout': go.Layout(
                                        height=400,
                                        width=1200,
                                    )


                            }),
                            ], 
                        ),  width={"offset":2},
                    ),
                    ),
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                            [
                                dcc.Graph(
                                    id="Survival Prediction",
                                    figure=go.Figure(
                                        data=[trace1s,trace2s],
                                    layout=go.Layout(
                                        title="Patient's Survival Rates Prediction (%)"
                                    ),
                                    ),
                                ),
                            ], 
                            ),  width={"offset":2},
                        ),
                    ),
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                            [ 
                                html.Img(
                                    src=app.get_asset_url('waffle-chart-km.png'),
                                    id="waffle-1",
                                    style={
                                        "height": "200px",
                                        "width": "1000px",
                                        "margin-bottom": "25px",
                                        "margin-left":"px"
                                    },
                                ),
                                
                                html.P('Out of a 100 random women, X will be dead within 10 years... '),
                                html.Br(),
                                html.Img(
                                    src=dash_app.get_asset_url('waffle-chart-survived.png'),
                                    id="waffle-2",
                                    style={
                                        "height": "200px",
                                        "width": "1000px",
                                        "margin-bottom": "25px",
                                        "margin-left":"px"
                                    },
                                ),
                                html.Div([
                                    html.P("Out of 100 random breast cancer patients, X will survive within the 10 year time period.")
                                ]), 
                                
                            ], 
                            ),  width={"offset":2},
                        ), 
                    ),        
                ]
            )  
            return survival_layout, patient_button, patient_graphs
        elif pathname =="/survival/doctor":
            cookies = session['received']
            cookies = cookies.decode("utf-8")
            group =cookies.split(",")
            # # # cookie = html.H1(cookies)
            # print(str(group)[3:10], "group 1")
            km = pd.read_csv('data\\kaplan_meier_by_group.csv')
            g_os = km.loc[km['class_label'] == 'OS']
            os = g_os.loc[g_os['group_label'] == str(group)[3:10]]
            g_dfs = km.loc[km['class_label'] == 'DFS' ]
            dfs = g_dfs.loc[g_dfs['group_label'] == str(group)[3:10]]
            g_css = km.loc[km['class_label'] == 'CSS' ]
            css = g_css.loc[g_css['group_label'] == str(group)[3:10]]
            # #ok = json_normalize(cookies)
            # #ok = pd.read_json(cookies)
            # ok = pd.DataFrame(cookies)
            # print(ok)
            patient = pd.read_csv("..\\middleWomen\\patient_new.csv")

            x = patient["x"]
            y = patient["y"]

            surv4 = (go.Scatter(x=x/365.25, y=y*100, name="hv",
                                line_shape='hv'))

            #overall survival Kaplan Meier chart
            x = os["time"]
            y = os["estimate"]
            lower = os["lower"]
            upper = os["upper"]
            km_upper = go.Scatter(x=x, y=y*100,
                fill=None,
                mode='lines',
                line_color='indigo',
                name='Overall Survival',
            )

            km_lower = go.Scatter( x=x,
                y=upper*100,
                fill='tonexty', # fill area between trace0 and trace1
                mode='lines', 
                line_color='lightblue',
                name="{}".format('95% Upper CI'),
            )
            km = go.Scatter( x=x,
                y=lower*100,
                fill='tonexty', # fill area between trace0 and trace1
                mode='lines', 
                line_color='lightblue',
                name="{}".format("95% Lower CI"),
            )

            #dfs Kaplan Meier chart 
            dfs_x = dfs["time"]
            dfs_y = dfs["estimate"]
            dfs_lower = dfs["lower"]
            dfs_upper = dfs["upper"]
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

            #css Kaplan Meier chart
            css_x = css["time"]
            css_y = css["estimate"]
            css_lower = css["lower"]
            css_upper = css["upper"]
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

            #doctor's graphs
            doctor_graphs =  html.Div(
                [
                    html.Br(),
                    html.Br(),
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Survival Prediction",
                                        figure=go.Figure(
                                                            data=[km_upper,km_lower,km],
                                                            layout=go.Layout(
                                                                title="Patient's Overal Survival Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,  
                                                                xaxis_range=(0, 10),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'}, 
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                            dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Kaplan Meier",
                                        figure=go.Figure(
                                                            data=[dfs_km_upper,dfs_km_lower,dfs_km],
                                                            layout=go.Layout(
                                                                title="Patient's Disease-Free Survival Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,
                                                                xaxis_range=(0, 10),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'}, 
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Kaplan Meier",
                                        figure=go.Figure(
                                                            data=[css_km_upper,css_km_lower,css_km],
                                                            layout=go.Layout(
                                                                title="Patient's Cancer Specific Survival Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,
                                                                xaxis_range=(0, 10),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'}, 
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="Kaplan Meier",
                                        figure=go.Figure(
                                                            data=[surv4],
                                                            layout=go.Layout(
                                                                title="Patient's Predicted Kaplan Meier Chart",
                                                                height=400,
                                                                width=1200,
                                                                xaxis_range=(0, 10),
                                                                yaxis_range=(0, 100),
                                                                xaxis = {'title': 'Year'},
                                                                yaxis = {'title': 'Percentage of Survival'}, 
                                                                hovermode= "closest",
                                                            ),
                                                        ),
                                    ),
                                ],
                            ), width = {"offset": 2}
                        )
                    ),
                    
                ]
            )
            
            return survival_layout, doctor_button, doctor_graphs

    @dash_app.callback(
        [
            dash.dependencies.Output('age-distribution-hist', 'figure'),
            dash.dependencies.Output('alive_dead_bar','figure'),
            dash.dependencies.Output('er_pr_chart','figure')
        ],
        [
            dash.dependencies.Input('age_slider', 'value'),
            dash.dependencies.Input('tnm_select','value'),
            dash.dependencies.Input('er_select','value'),
            dash.dependencies.Input('pr_select','value'),
            dash.dependencies.Input('her2_select','value')
        ]
    )

    
    def update_clinical(age_slider,tnm_select, er_select, pr_select, her2_select):
        #Slice Df according to inputs in filter
        df = filter_df_all(clinical, age_slider[0], age_slider[1] , tnm_select, er_select, pr_select, her2_select)
        # df = clinical[(clinical['Age_@_Dx'] > age_slider[0]) & (clinical['Age_@_Dx'] < age_slider[1]) & (clinical['ER'] == er_select) ]

        #editing alive vs dead bar chart - overwrite initial version
        cause = df['cause_of_death']
        dcdict_new = calPercent(df,cause,True,"Alive")
        # print(calPercent(df, df['cause_of_death'], True, "Alive"))
        dcdict = rename_keys(dcdict_new,\
                                ['Alive', 'Dead- Breast Cancer', 'Dead- Others', 'Dead- Unknown'])
        print(dcdict_new)
        #Overwrite initial tnm chart
        # tnm_df = filter_df_wo_stage(clinical,age_slider[0], age_slider[1] , er_select, pr_select, her2_select )
        
        # fdict = generate_tnm_chart_data(tnm_df, cause, death_cause_dict_old)
        # print(fdict)

        #Overwrite original chart data with sliced dataset according to filters
        epr_df = filter_df_epr_chart(clinical, age_slider[0], age_slider[1] , tnm_select)
        er_dict = generate_epr_chart_data(epr_df)
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
                    xaxis = {'title': 'Cause of Death'},
                    yaxis = {'title': 'Percentage of Patients'},
                    hovermode='closest'
            )
        }

        # figure3 ={
        #         'data': [
        #                 dict(
        #                     x= fdict['Alive'],
        #                     y= clinical['Stage'].dropna().unique(),
        #                     type='bar',
        #                     name='Alive',
        #                     orientation='h',
        #                     marker=dict(
        #                     color='lightgreen',
        #                     line=dict(color='lightgreen', width=3))
        #                 ),
        #                 dict(
        #                     x= fdict['breast cancer related'],
        #                     y= clinical['Stage'].dropna().unique(),
        #                     type='bar',
        #                     name='Dead- Breast cancer related',
        #                     orientation='h',
        #                     marker=dict(
        #                     color='lightcoral',
        #                     line=dict(color='lightcoral', width=3)
        #                     )
        #                 ),
        #                 dict(
        #                     x= fdict['n'],
        #                     y= clinical['Stage'].dropna().unique(),
        #                     type='bar',
        #                     name='Dead',
        #                     orientation='h',
        #                     marker=dict(
        #                     color='indianred',
        #                     line=dict(color='indianred', width=3)
        #                     )
        #                 ),
        #                 dict(
        #                     x= fdict['unknown'],
        #                     y= clinical['Stage'].dropna().unique(),
        #                     type='bar',
        #                     name='Unknown',
        #                     orientation='h',
        #                     marker=dict(
        #                     color='lightslategrey',
        #                     line=dict(color='lightslategrey', width=3)
        #                     )
        #                 )
        #             ],
        #             'layout': go.Layout(
        #                 title = "TNM Stage Alive Vs Dead",
        #                 xaxis = {'title': 'Percentage of Patients'},
        #                 yaxis = {'title': 'Cancer Stages'},
        #                 hovermode='closest',
        #                 barmode='stack',
        #             )
        #         }
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
        return figure, figure2, figure4


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
                                xaxis = {'title': 'Patient ID'},
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