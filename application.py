import streamlit as st
import os
from CNN_Classifier.utils.common import decode_image
from CNN_Classifier.pipeline.prediction import PredictionPipeline
import pandas as pd

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

obj = ClientApp()

st.title('Kindney Disease Classification Using Deep Learning')
st.divider()
st.subheader("Upload your image of kidney ct-scan to cheack for any disease")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    with open(obj.filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
        f.close()
    prediction, result_raw = obj.classifier.predict()
    if(prediction=='Failed'):
        st.write("Something went wrong")
    else:
        st.write("You have ",prediction," kidney")
        df = pd.DataFrame(data=result_raw, columns=['Cyst', 'Normal', 'Stone', 'Tumor'], index=None)
        st.write("Probablity for each case : ")
        st.dataframe(df)



