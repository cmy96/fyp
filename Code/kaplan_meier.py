import pandas as pd
import numpy as np
import scipy.stats
import pickle
import fnmatch
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="colorblind", color_codes=True)

from survive import datasets
from survive import SurvivalData
from survive import KaplanMeier, Breslow, NelsonAalen
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from pprint import pprint
pd.set_option('display.width', None)
pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)




# ========== KAPLAN MEIER FUNCTIONS ==========
def generate_kaplan_meier_with_filters(filters_dict, input_df, survival_type = "none"):
    """
    This is the main Kaplan Meier function to be called by the application. It builds a dataframe from csv and filters accordingly to generate the kaplan meier chart. 
    """

    input_df = input_df
    #load data from clinical if no df was given.
    if survival_type == "none":
        survival_type = "OS"

    # build survival obj:
    survival_obj = build_surv_obj(survival_type="OS" ,filters_dict = filters_dict, input_df=input_df)
    km = KaplanMeier()
    km.fit(survival_obj)

    # Plot Curve
    plt.figure(figsize=(10, 6))
    km.plot()
    plt.show()
    plt.close()

    # generate output df:
    output_df = KM_to_df(km)
    # display(output_df.head())

    return output_df


def kaplan_meier_drop_by_index(X,indexes):
    """
    helper function to drop rows of dataframe and return new dataframe without those rows with indexes resetted
    """
    X = X.drop(indexes)
    X = X.reset_index().drop(columns="index")
    return(X)

def kaplan_meier_load_clinical_df(dropCol,FILE_FOLDER = "C:\\SMU_v2\\"):
    '''
    function to read the pkl from from datasource
        1. Remove dx_date that is NULL.
        2. Drop all rows where crucial fields for X_features are NULL.
        3. Convert Date columns into datetime format
        4. Derive OS, CSS, DFS days based on dx_date
        5. Create status column to indicate if the patient is dead or alive base on if death_age exists
    '''
    df = pd.read_pickle(FILE_FOLDER + "clinical_output.pkl").reset_index().drop(columns="index")
    to_drop = df[df['dx_date']=="NA"].index
    df = kaplan_meier_drop_by_index(df,to_drop)

    df.drop(columns=dropCol,inplace = True)

    # drop all rows where dates are null
    df.dropna(axis=0,\
                    subset=['Date_for_DFS','Date_for_OS','Date_for_CSS','dx_date','Age_@_Dx'],\
                    inplace=True)
    
    # convert all datetime in dataframe into dateime format for processing
    df["Date_for_DFS"] = pd.to_datetime(df["Date_for_DFS"])
    df["Date_for_OS"] = pd.to_datetime(df["Date_for_OS"])
    df["Date_for_CSS"] = pd.to_datetime(df["Date_for_CSS"])
    df["dx_date"] = pd.to_datetime(df["dx_date"])
    df['last_seen']= pd.to_datetime(df["dx_date"])
    df['dob']= pd.to_datetime(df["dx_date"])

    # calculate in days
    df["DFS_days"] = (df["Date_for_DFS"] - df['dx_date'] )/np.timedelta64(1, 'D')
    df["OS_days"] = (df["Date_for_OS"] - df['dx_date'] )/np.timedelta64(1, 'D')
    df["CSS_days"] = (df["Date_for_CSS"] - df['dx_date'] )/np.timedelta64(1, 'D')

    # alive or dead
    df['status'] = np.where(df['Count_as_OS'] == "dead", False, True)

    # convert all er pr her2 to lower
    df["ER"] = df["ER"].apply(lambda x: x.lower() if x.isalpha() else x)
    df["PR"] = df["PR"].apply(lambda x: x.lower() if x.isalpha() == False else x)
    df["Her2"] = df["Her2"].apply(lambda x: x.lower() if x.isalpha() == False else x)

    return df

def build_surv_obj(survival_type, input_df, filters_dict):
    
    """
    This function builds the survival object to be processed by kaplan meier model to return kaplan meier df
    it first filters the full df by the following columns:
    1. Age_@_Dx
    2. Race
    3. T
    4. N
    5. M
    6. ER
    7. PR
    8. Her2

    then it takes the results and builds the survival model
    """

    # filter by filters dict:
    age_lower = filters_dict["age_lower"]
    age_upper = filters_dict["age_upper"]
    race = filters_dict["Race"]
    t_stage = filters_dict["T"]
    n_stage = filters_dict["N"]
    m_stage = filters_dict["M"]
    er = filters_dict["ER"]
    pr = filters_dict["PR"]
    her_2 = filters_dict["Her2"]

    print("This is the input df shape before filters ", input_df.shape)

    # filter by age range
    temp_df = input_df[(input_df["Age_@_Dx"] >= age_lower) & (input_df["Age_@_Dx"] <= age_upper)]

    # filter by race if race was selected
    if race == "all":
        print("race was selected as 'all'")
    else:
        temp_df = input_df[(input_df["Race"] == race)]
    

    # filter by TNM
    temp_df = temp_df[(temp_df["T"] == t_stage) & (temp_df["N"] == n_stage) & (temp_df["M"] == m_stage)]
    # TODO: FIX THIS BUG filter by er pr her2 (STILL BUGGY)
    temp_df = temp_df[(temp_df["ER"] == er) & (temp_df["PR"] == pr) & (temp_df["Her2"] == her_2)]

    print("This is the input df shape AFTER filters ", temp_df.shape)
    
    survival_df = temp_df
    survival_type = str(survival_type)
    
    Time_df = survival_df.loc[:,[survival_type + "_days"]]
    Time_df[survival_type + "_years"] = Time_df[survival_type + "_days"]/365.25
    Time_df["status"] = survival_df["Count_as_" + survival_type].apply(lambda status: 0 if status in "nN" else 1)
    Time_df["check"] = survival_df["Count_as_" + survival_type]

    return SurvivalData(time= (survival_type+ "_years"), status="status", data=Time_df)

def KM_to_df(KM_object):
    
    # Process the summary as string    
    summary_lines_list = str(KM_object.summary).split("\n")
    
    header = ["time", "events", "at_risk",  "estimate",  "std_error",  "95%_CI_lower",  "95%_CI_upper"]
    rows = summary_lines_list[6:]
    
    row_values = []
    
    for row in rows:
        
        elements = row.split(" ")
        tmp = []
        for element in elements:
            if element.isnumeric() or ("." in element):
                tmp.append(element)
                
        row_values.append(tmp)
        
    #Build df
    output_df = pd.DataFrame()
    temp_df = pd.DataFrame(row_values, columns=header)
    output_df["time"] = temp_df["time"]
    output_df["estimate"] = temp_df["estimate"]
    output_df["lower"] = temp_df["95%_CI_lower"]
    output_df["upper"] = temp_df["95%_CI_upper"]
                
    return output_df




