import streamlit as st
import numpy as np
import joblib
from prediction import predict

st.title('Parkinsons Disease Prediction')
st.markdown('Please enter the following voice frequencies in order to predict if you have parkinsons disease or not')
col1,col2=st.columns(2)

with col1:
    MDVPFo=st.slider('MDVP:Fo(Hz)',0.0,1.0,0.1)
    MDVPFhi=st.slider('MDVP:Fhi(Hz)',0.0,1.0,0.1)
    MDVPFlo=st.slider('MDVP:Flo(Hz)',0.0,1.0,0.1)
    MDVPJitter=st.slider('MDVP:Jitter(%)',0.0,1.0,0.1)
    MDVPJitterabs=st.slider('MDVP:Jitter(Abs)',0.0,1.0,0.1)
    MDVPRap=st.slider('MDVP:RAP',0.0,1.0,0.1)
    MDVPppq=st.slider('MDVP:PPQ',0.0,1.0,0.1)
    JitterDDP=st.slider('Jitter:DDP',0.0,1.0,0.1)
    MDVPshimmer=st.slider('MDVP:Shimmer',0.0,1.0,0.1)
    MDVPshimmerdb=st.slider( 'MDVP:Shimmer(dB)',0.0,1.0,0.1)
    ShimmerAPQ3=st.slider('Shimmer:APQ3',0.0,1.0,0.1)
    ShimmerAPQ5=st.slider('Shimmer:APQ5',0.0,1.0,0.1)
    
with col2:
    MDVPapq=st.slider('MDVP:APQ',0.0,1.0,0.1)
    Shimmerdda=st.slider('Shimmer:DDA',0.0,1.0,0.1)
    nhr=st.slider('NHR',0.0,1.0,0.1)
    hnr=st.slider('HNR',0.0,1.0,0.1)
    rpde=st.slider('RPDE',0.0,1.0,0.1)
    dfa=st.slider('DFA',0.0,1.0,0.1)
    spread1=st.slider('spread1',0.0,1.0,0.1)
    spread2=st.slider('spread2',0.0,1.0,0.1)
    D2=st.slider('D2',0.0,1.0,0.1)
    ppe=st.slider('PPE',0.0,1.0,0.1)

st.text('')
st.text('')

if st.button('Predict Parkinsons disease'):
    result=predict(np.array([[MDVPFo, MDVPFhi,MDVPFlo,MDVPJitter,MDVPJitterabs,MDVPRap,MDVPppq,JitterDDP,MDVPshimmer,MDVPshimmerdb,ShimmerAPQ3, ShimmerAPQ5,
                              MDVPapq,Shimmerdda,nhr,hnr,rpde,dfa,spread1,spread2,D2,ppe]]))
    if (result[0]==0):
        st.text('You donot have parkinsons')
    else:
        st.text('You have parkinsons')
    
    
    
st.text('')
st.text('')

st.markdown('Parkinsons disease prediction @2024')