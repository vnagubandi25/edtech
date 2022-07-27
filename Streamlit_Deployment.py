import pickle
import streamlit as st
from pycaret.regression import *
import pandas as pd
 

 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def predict(
       Program_mode,
       Course_level, 
       Course_duration, 
       Certification_provided,
       Practice_assignments, 
       Doubt_sessions,
       Internship_provided,
       Placement_assistance,
       Course_rating,
       Recorded_videos,
       Trainer_level
       ):


    model = load_model('edtech')

    data = pd.DataFrame([[
              Program_mode, 
              Course_level, 
              Course_duration, 
              Certification_provided, 
              Practice_assignments, 
              Doubt_sessions,
              Internship_provided, 
              Placement_assistance,
              Recorded_videos,
              Trainer_level,
              (np.exp(Course_rating,dtype=np.float32))
       ]])


    data.columns =['Program_mode', 'Course_level', 'Course_duration',
              'Certification_provided', 'Practice_assignments', 'Doubt_Sessions',
              'Internship_provided ', 'Placement_assistance', 'Recorded_videos ',
              'Trainer_level', 'Log_rating']

    predictions = predict_model(model, data=data) 
    return(np.exp(int(predictions['Label'][0])))
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Program_mode = st.selectbox('Program Mode',("online","offline","both"))
    Course_level = st.selectbox('Course Level',("low","medium","high"))
    Course_duration = st.number_input("Duration of Course")
    Certification_provided = st.selectbox('Certification Provided?',("yes","no"))
    Practice_assignments = st.selectbox('Practice_assignments',("yes","no"))
    Doubt_Sessions = st.selectbox('Doubt_Sessions',("yes","no"))
    Internship_provided  = st.selectbox('Internship_provided',("yes","no"))
    Placement_assistance  = st.selectbox('Placement_assistance',("yes","no"))
    Recorded_videos  = st.selectbox('Recorded_videos',("yes","no"))
    Trainer_level  = st.selectbox('Internship_provided ?',("low","medium","high"))
    Log_rating = st.number_input("Rating of Course")
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = predict(       Program_mode,
       Course_level, 
       Course_duration, 
       Certification_provided,
       Practice_assignments, 
       Doubt_Sessions,
       Internship_provided,
       Placement_assistance,
       Log_rating,
       Recorded_videos,
       Trainer_level) 
        st.success('Your loan is {}'.format(result))
     
if __name__=='__main__': 
    main()

