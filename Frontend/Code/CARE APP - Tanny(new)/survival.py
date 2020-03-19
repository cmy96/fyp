import tensorflow as tf

model = tf.keras.models.load_model('C:\\SMU_v2\\ann\\model_layer1_10y.h5' ,custom_objects={'leaky_relu': tf.nn.leaky_relu})

pred = pd.DataFrame(model.predict(X_test))