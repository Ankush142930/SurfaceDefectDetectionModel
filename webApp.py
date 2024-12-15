import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

st.header('Surface Defect Detection CNN Model')
defect_name = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

model = load_model('Surface_Defect_Detection_Model.h5')

def classify_defect(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image is of ' + defect_name[np.argmax(result)] + ' defect with a score of '+ str(np.max(result)*100)
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width = 200)

    st.markdown(classify_defect(uploaded_file))