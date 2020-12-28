from sksurv.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import pickle
import numpy as np
import pandas as pd
import ast
import json
import tensorflow as tf

# Load encoder to OHE new raw data for prediction
def loadOHE(df,OHE_LOCATION = "C:\\SMU_v2\\OHE\\", name=""):
    '''
    load enconder to OHE new raw data for prediction
    '''
    with open( "{}{}{}".format(OHE_LOCATION, name, '_encoder.pickle'), 'rb') as f:
        enc = pickle.load(f) 
    
    #type case object to category
    typeCastList = list(df.select_dtypes(include=[object]).columns)
    df[typeCastList] = df[typeCastList].astype("category")
    OHE_New_Data = enc.transform(df)
    
    return OHE_New_Data

# Get output of survival model and load into doctor predicted survival charts
def get_patient_prediction(raw_data,group):
    
    if group == 1:
        
        model = tf.keras.models.load_model('C:\\SMU_v2\\ann\\model_group1_10y.h5',\
                                           compile=False,
                                           custom_objects={'leaky_relu': tf.nn.leaky_relu})
        pred_10y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group1_5y.h5')
        pred_5y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group1_2y.h5')
        pred_2y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group1_1y.h5')
        pred_1y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group1_6m.h5')
        pred_6m = model.predict(raw_data)[0][0]

    elif group == 2:
 
        model = tf.keras.models.load_model('C:\\SMU_v2\\ann\\model_group2_10y.h5',\
                                           compile=False,\
                                           custom_objects={'leaky_relu': tf.nn.leaky_relu})
        pred_10y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group2_5y.h5')
        pred_5y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group2_2y.h5')
        pred_2y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group2_1y.h5')
        pred_1y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group2_6m.h5')
        pred_6m = model.predict(raw_data)[0][0]

    elif group == 3:
        model = tf.keras.models.load_model('C:\\SMU_v2\\ann\\model_group3_10y.h5',\
                                           compile=False,\
                                           custom_objects={'leaky_relu': tf.nn.leaky_relu})
        pred_10y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group3_5y.h5')
        pred_5y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group3_2y.h5')
        pred_2y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group3_1y.h5')
        pred_1y = model.predict(raw_data)[0][0]
        model.load_weights('C:\\SMU_v2\\ann\\model_group3_6m.h5')
        pred_6m = model.predict(raw_data)[0][0]

    to_return = pd.DataFrame([[pred_6m,pred_1y,pred_2y,pred_5y,pred_10y]],columns = ["6m","1y","2y","5y","10y"])
    return to_return

# Prepare data for survival table
def survivalTable(modelName, raw_data):
    '''
    Calculate survival rate in years of interest
    '''
    interval = list([0.5,1,2,5,10])
        
    model = joblib.load('C:\\SMU_v2\\Model_folder\\{}.pkl'.format(modelName))

    surv = model.predict_survival_function(raw_data)
    
    dic = {}
    
    for i, s in enumerate(surv):
        x = model.event_times_
        y = s
    graphaxis = pd.DataFrame({'x':x,'y':y}, columns = ['x','y'])
    for i in interval:
        result = np.where(x > (365.25*(i+1)))[0][0]
        dic[str(i) + " years"] = [y[result]]
    dic = pd.DataFrame.from_dict(dic)
    return dic,graphaxis

# Categorize patient inputs into survival model's groupings
def categorize(input):

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

        group = "group 1"
    # group 2
    elif raw['stage'] == 'dcis/lcis non-invasive':
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

        raw_data = {
                    'ER': [raw['ER']],\
                    'PR': [raw['PR']],\
                    'Her2': [raw['Her2']],\
                    'Size': [size],\
                    'Age_@_Dx': [int(raw['Age_@_Dx'])],\
                    'diff': [raw['diff']]
                   }
        MTU = 'group 2_layer 1_rsf'

        group = "group 2"
    else:
        # group 3
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

        group = "group 3"

    for k,v in raw_data.items():
        if str(v[0]).isalpha():
            raw_data[k] = v[0].lower()

    raw_data = pd.DataFrame.from_dict(raw_data)
    raw_data = loadOHE(raw_data,OHE_LOCATION = "C:\\SMU_v2\\OHE\\", name=MTU[:-4])

    z,DF = survivalTable(MTU,raw_data)  
    to_return = get_patient_prediction(raw_data,int(group[-1]))

    return z, DF, group, to_return
