import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("/home/ssahoo/Desktop/SWM_FINAL/model_final_1.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(150,150))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


st.title("Waste Classification")
test_image = st.file_uploader("Select an Image")
if(st.button("Show Image")):
    if(test_image):
        st.image(test_image,use_column_width=True)
    else:
        st.write("Please select an image")

if (st.button("Predict")):
    if (test_image):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)

            class_name = ['organic', 'recyclable']
            st.success("Model is predicting. It is {}".format(class_name[result_index]))
    else:
        st.write("Please select an image")