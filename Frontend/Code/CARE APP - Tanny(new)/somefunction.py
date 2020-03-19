from sksurv.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import pickle
import numpy as np
import pandas as pd
import ast
import json

def survivalTable(modelName, raw_data):
    '''
    Calculate survival rate in years of interest
    '''
    OHE_LOCATION = "C:\\SMU_v2\\OHE\\"  
    interval = list([0.5,1,2,5,10])

    for k,v in raw_data.items():
        if str(v[0]).isalpha():
            raw_data[k] = v[0].lower()
        
    raw_data = pd.DataFrame.from_dict(raw_data)
    
    model = joblib.load('..\\..\\..\\Code\\Model_folder\\{}.pkl'.format(modelName))

    with open( "{}{}{}".format(OHE_LOCATION, modelName[:-4], '_encoder.pickle'), 'rb') as f:
            enc = pickle.load(f) 
        
    #type case object to category
    typeCastList = list(raw_data.select_dtypes(include=[object]).columns)
    raw_data[typeCastList] = raw_data[typeCastList].astype("category")
    data = enc.transform(raw_data)

    surv = model.predict_survival_function(data)
    
    dic = {}
    
    for i, s in enumerate(surv):
        x = model.event_times_
        y = s
    graphaxis = pd.DataFrame({'x':x,'y':y}, columns = ['x','y'])
    for i in interval:
        result = np.where(x > (365.25*(i+1)))[0][0]
        dic[i] = y[result]
    return dic,graphaxis

def haha(input):

    # data2 = {'Cancer drugs': 268941532, 'Hospital care': 96600809, 'Investigations': 247463280, 'Other drugs': 37384671, 
    # 'Other surgeries': 137540609, 'Others': 67517445, 'Professional services': 87810977, 'Radiation therapy': 78311175, 
    # 'Surgical procedures': 53562182}
    # df = pd.DataFrame(data2, index=[0])
    
    edit = str(input)
    edit = edit[2:len(edit)-1]

    raw = ast.literal_eval(edit)

    # group 1
    if raw['stage'] == 'stage 4':

        raw_data = {
                    'ER': [raw['ER']],\
                    'PR': [raw['PR']],\
                    'Her2': [raw['Her2']],\
                    'size_precise': [float(raw['size_precise'])],\
                    'nodespos': [int(raw['nodespos'])],\
                    'Age_@_Dx': [int(raw['Age_@_Dx'])],\
                    'diff': [raw['diff']]
                }
        MTU = 'group 1_layer 4_rsf'
        print("!!!!!!!!!!!!!!!!!!!hi")
    # group 2
    elif raw['stage'] == 'dcis/lcis non invasive':
        if float(raw['size_precise']) <= 1.0:
            size = "0 - 1 cm"
        elif float(raw['size_precise']) <= 2.0:
            size = "1.01 - 2 cm"
        elif float(raw['size_precise']) <= 3.0:
            size = "2.01 - 3 cm"
        elif float(raw['size_precise']) <= 4.0:
            size = "3.01 to 4 cm"
        elif float(raw['size_precise']) <= 5.0:
            size = "4.01 to 5 cm"
        elif float(raw['size_precise']) >= 5.0:
            size = "> 5 cm"     
        #---- to ask munyee !! ---
        
        # else: 
        #     size = "unknown"    

        raw_data = {
                    'ER': [raw['ER']],\
                    'PR': [raw['PR']],\
                    'Her2': [raw['Her2']],\
                    'Size': [size],\
                    'Age_@_Dx': [int(raw['Age_@_Dx'])],\
                    'diff': [raw['diff']]
                   }
        MTU = 'group 2_layer 1_rsf'
        print("!!!!!!!!!!!!!!!!!!!",size)
    else:
        # # group 3
        raw_data = {
                    'ER': [raw['ER']],\
                    'PR': [raw['PR']],\
                    'Her2': [raw['Her2']],\
                    'size_precise': [float(raw['size_precise'])],\
                    'nodespos': [int(raw['nodespos'])],\
                    'Age_@_Dx': [int(raw['Age_@_Dx'])],\
                    'T':[raw['tstage']],\
                    'N': [raw['nstage']],\
                    'M': [raw['mStage']],
                   }
        MTU = 'group 3_layer 5_rsf'
        print("!!!!!!!!!!!!!!!!!!!bye")

    z,DF = survivalTable(MTU,raw_data)  
  
    return DF