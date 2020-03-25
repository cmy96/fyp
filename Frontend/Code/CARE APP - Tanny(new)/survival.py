import tensorflow as tf

model = tf.keras.models.load_model('C:\\SMU_v2\\ann\\model_group1_10y.h5' ,custom_objects={'leaky_relu': tf.nn.leaky_relu})

X_test = {
            'ER': ['positive'],\
            'PR': ['positive'],\
            'Her2': ['negative'],\
            'size_precise': [1.3],\
            'nodespos': [0],\
            'Age_@_Dx': [21],\
            'diff': ['grade 3']
           }


pred = pd.DataFrame(model.predict(X_test))

print(pred) 