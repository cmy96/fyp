import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
import pathlib
import plotly.graph_objects as go
import operator
import logging
from logging.handlers import RotatingFileHandler

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data
clinical = pd.read_csv(DATA_PATH.joinpath("clinical.csv"))

#Get unique vals of each col for filters
tnm_list = clinical["TNM_Stage"].dropna().unique()
ER_list = clinical['ER'].dropna().unique()
pr_list = clinical['PR'].dropna().unique()
Her2_list = clinical['Her2'].dropna().unique()
clinical['Age_@_Dx'] = clinical['Age_@_Dx'].astype(int)

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
                        ['Alive', 'Dead- Breast Cancer', 'Dead- Others', 'Dead- Unknown'])


#Binning of diagnosed age - Tanny
bins = np.arange(clinical['Age_@_Dx'].min(),clinical['Age_@_Dx'].max() + 4, 4)
clinical['binned'] = np.searchsorted(bins, clinical['Age_@_Dx'].values)
age_bin_count = clinical.groupby(pd.cut(clinical['Age_@_Dx'], bins=19, precision = 0, right = False)).size()


    
def generate_tnm_chart_data(df, dc):
    # for every stage in TNM_stage, it becomes the key for the MAIN dictionary
    #assuming no nan data
    TNM = df['TNM_Stage']
    TNM_Stage = TNM.dropna().unique()
    TNM_dict={}
    for stage in TNM_Stage:
        status_dict = {}
        # tnm_death_cause_dict = calPercent(df,dc,True,"Alive")
        

        #every stage in tnm_stage will have a dictionary that holds death_status as key and number of death as value
        for life_status in death_cause_dict_old.keys():
            tmp = df[['TNM_Stage','cause_of_death']]
            condition1 = df[TNM==stage]
            condition2 = tmp[(tmp['cause_of_death']==life_status) & (tmp['TNM_Stage']==stage)]
            condition3 = tmp[(tmp['cause_of_death'].isnull()) & (tmp['TNM_Stage'] == stage)]

            if life_status == "Alive":
                if len(condition1) > 0 or len(condition2) > 0:
                    NumRecord = len(condition3)/len(condition1)*100
            else:
                if len(condition1) > 0 or len(condition2) > 0:
                    NumRecord  = len(condition2)/len(condition1)*100
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
    print(finalized_dict)
    return TNM_dict, finalized_dict


def generate_epr_chart_data(df):
    ER_dict = {}
    ERlist = list(['positive','negative','equivocal','unknown'])
    PRlist = list(['positive','negative','equivocal','unknown'])

    for status_er in ERlist:
        status_dict = {}
        for status_pr in PRlist:
            tmp = df[['ER','PR']]
            if len(df[df['ER']==status_er]) > 0 or len(tmp[(tmp['ER']==status_er) & (tmp['PR']==status_pr)]) > 0:
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

    return er_finalized_dict

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
    print(output)
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
                value=tnm_list[2],
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

TNM_dict, finalized_dict = generate_tnm_chart_data(clinical, death_cause)
er_finalized_dict = generate_epr_chart_data(clinical)

app.layout = html.Div(
    
    id="app-container",
    children=[
    dcc.Store(id='session', storage_type='session'),
    html.Div(
        id="banner",
        className="banner",
        children=[
            # html.Img(src=app.get_asset_url('logo.png'))
            ],
    ),
    #CONTROL PANEL
    html.Div(
        id="controls",
        className="four columns",
        children=[description_card(), generate_controls()]
        + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
    ),
    #Main Dashboard
    html.Div([
      
        html.Div(
            id="main-dash",
            className="eight columns",
            children=[
                html.Div(
                    id="main-dash-card",
                    className = 'four columns',
                    children=[
                        html.B("Age Distribution"),
                        dcc.Graph(id="age-distribution-hist",
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
                        'layout': {
                            'title': "Patient's Diagnosed Age Distribution",
                            'xaxis': "Diagnosis Age",
                            'yaxis':'Percentage of Patients'
                        }
                        }),
                    ]
                ),
                html.Div(
                    id="alive-dead-card",
                    className = "four columns",
                    children=[
                        html.B("Proportion of Alive & Dead patients"),
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
                    ]
                )
                
                    ]
                ),
                html.Div([
                    html.Div(
                        id="tnm-stage-card",
                        children=[
                            html.B('TNM Stage Alive Vs Dead'),
                            dcc.Graph(
                                id="tnm-stage-stacked-bar",
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
                                        showlegend=False
                                    )
                                }
                            ),
                            html.Br(),
                            html.B("ER VS PR"),
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
                            )
                        ],className = 'four columns'
                    )
                ],className='eight columns'
                )

    ])


]) #end


#Callbacks

app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output('output-clientside','children'),
    [Input('age_slider','value')]
)

@app.callback(
    [
        dash.dependencies.Output('age-distribution-hist', 'figure'),
        dash.dependencies.Output('alive_dead_bar','figure'),
        dash.dependencies.Output('tnm-stage-stacked-bar', 'figure'),
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
def update_all(age_slider,tnm_select, er_select, pr_select, her2_select):
    #Slice Df according to inputs in filter
    df = filter_df(clinical, age_slider[0], age_slider[1] , tnm_select, er_select, pr_select, her2_select)
    # df = clinical[(clinical['Age_@_Dx'] > age_slider[0]) & (clinical['Age_@_Dx'] < age_slider[1]) & (clinical['ER'] == er_select) ]

    #editing alive vs dead bar chart - overwrite initial version
    cause = df['cause_of_death']
    dcdict_new = calPercent(df,cause,True,"Alive")
    dcdict = rename_keys(dcdict_new,\
                            ['Alive', 'Dead- Breast Cancer', 'Dead- Others', 'Dead- Unknown'])

    #Overwrite initial tnm chart
    tdict, fdict = generate_tnm_chart_data(df, cause)

    #Overwrite original chart data with sliced dataset according to filters
    er_dict = generate_epr_chart_data(df)
    
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
         'layout': layout
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
                        x= list(fdict['Alive']),
                        y= list(tdict.keys()),
                        type='bar',
                        name='Alive',
                        orientation='h',
                        marker=dict(
                        color='lightgreen',
                        line=dict(color='lightgreen', width=3)
                        
                        )
                    ),
                    dict(
                        x= list(fdict['breast cancer related']),
                        y= list(tdict.keys()),
                        type='bar',
                        name='Dead- Breast cancer related',
                        orientation='h',
                        marker=dict(
                        color='lightcoral',
                        line=dict(color='lightcoral', width=3)
                        )
                    ),
                    dict(
                        x= list(fdict['n']),
                        y= list(tdict.keys()),
                        type='bar',
                        name='Dead',
                        orientation='h',
                        marker=dict(
                        color='indianred',
                        line=dict(color='indianred', width=3)
                        )
                    ),
                    dict(
                        x= list(fdict['unknown']),
                        y= list(tdict.keys()),
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


if __name__ == "__main__":
    # handler = RotatingFileHandler('dash.log', maxBytes=10000, backupCount=1)
    # handler.setLevel(logging.INFO)
    # app.server.logger.addHandler(handler)
    app.run_server(debug=True, port=8060)