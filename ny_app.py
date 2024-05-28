# ---------------------LIBRARIES---------------------- #
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import warnings
import base64
from pandas.errors import SettingWithCopyWarning

# Graphics
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px
from plotly.subplots import make_subplots

# Interactive maps
import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import folium_static

# Prediction
import xgboost as xgb
import json
from joblib import dump, load

warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))


# ---------------------SITE CONFIGURATION---------------------- #
st.set_page_config(
    page_title="Airbnb: New York",
    page_icon="🗽",
    layout="centered", 
    initial_sidebar_state="collapsed", 
)


# ---------------------MENU---------------------- #

# Header image
st.image("images/ny_airbnb.png")

# Menu bar
page = option_menu(None, ["Home", "Airbnb info", "Reviews", "Price predictor"], 
    icons=["house", "pin-map", "table", "coin"], 
    default_index=0, orientation="horizontal",
    styles={
        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "padding": "0px", "--hover-color": "#eee"},
        "icon": {"margin": "auto", "display": "block"}  # Centered icons
    }
) 


# ---------------------LOAD DATA---------------------- #

# read data
@st.cache_data()
def load_data():
    df = pd.read_csv("datasets/df_short.csv")
    return df

# load data
df = load_data()


# ---------------------BACKGROUND IMAGE---------------------- #

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
     <style>
        .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local("images/ny_wallpaper_white.jpg")  


# ---------------------BODY---------------------- #

# PAGE 1-------------------------------------
if page == "Home":
    
    st.markdown("""
                ***RNew York, the city that never sleeps, has been a melting pot of cultures and religions for more than two centuries. Located in the state of New York, New York City is the largest city in the United States and a major global economic, cultural, and social hub. Today, it is a vibrant metropolis that combines the grandeur of its history with the vitality of a modern city, attracting millions of tourists every year.***
                """)

    st.markdown("""
             *In this context, the Airbnb accommodation market in New York plays an important role, offering visitors the opportunity to immerse themselves in the daily life of the city and experience it from a unique and personal perspective. This app analyzes the supply of Airbnb accommodation in New York, exploring its characteristics, trends, and impact on the city's real estate and tourism markets.*
             """)
    
    st.markdown("""
                
                The listings dataset has 39881 rows and 75 columns. In this case, we have created another record with 23 columns, which we are going to work on:

                1. ``host_since``: Date the host joined Airbnb.
                2. ``host_location``: Location of the host.
                3. ``host_response_rate``: Response rate of the host.
                4. ``host_is_superhost``: Indicates whether the host is a superhost or not.
                5. ``Neighbourhood``: City zone.
                6. ``neighbourhood_cleansed``: The neighbourhood as geocoded using the latitude and longitude against neighborhoods as defined by open or public digital shapefiles.
                7. ``neighbourhood_group_cleansed``: The neighbourhood group as geocoded using the latitude and longitude against neighborhoods as defined by open or public digital shapefiles.
                8. ``property_type``: Type of property.
                9. ``room_type``: Type of room offered.
                10. ``accommodates``: Maximum number of persons that can be accommodated in the property.
                11. ``beds``: The number of bed(s).      
                12. ``price``: Price per night from
                13. ``minimum_nights``: Minimum number of night stay for the listing.
                14. ``maximum_nights``: Maximum number of night stay for the listing.
                15. ``availability_30``: Availability of the property in the next 30 days.
                16. ``availability_60``: Availability of the property in the next 60 days.
                17. ``availability_90``: Availability of the property in the next 90 days.
                18. ``availability_365``: Availability of the property in the next year.
                19.	``number_of_reviews``: Total number of reviews.
                20.	``review_scores_rating``: Overall rating.
                21. ``review_scores_location``: Location score.
                22.	``reviews_per_month``: Average number of reviews received per month for a given property.
            
                """  )      
    st.write('------')                    
    st.markdown('### **Visualisation of the pre-processed dataframe:**')
    st.dataframe(df.head())
    st.write('Information about the code used can be found on my GitHub: https://github.com/MarcTorrentss/airbnb_project')



# PAGE 2-------------------------------------
elif page == "Airbnb info":

    st.markdown('Developing..')



# PAGE 3-------------------------------------
elif page == "Reviews":
        
    st.markdown('Developing..')



# PAGE 4-------------------------------------
elif page == "Price predictor":
    
    st.markdown("""
        <div style='text-align: center;'>
            <h1>Price prediction for Airbnb accommodation in New York</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Files upload    
    scaler = load('files/scaler.pkl') # Load the scaler
    model = load('models/model.pkl') # Load the best model trained
    
    with open("files/mapeo.json", "r") as json_file:
        # Loads the content of the JSON file into a dictionary
        encoder = json.load(json_file)

    # Create a dictionary of the neighbourhoods
    districts = {
    'Brooklyn': df[df['neighbourhood_group_cleansed'] == 'Brooklyn']['neighbourhood_cleansed'].unique().tolist(),
    'Manhattan': df[df['neighbourhood_group_cleansed'] == 'Manhattan']['neighbourhood_cleansed'].unique().tolist(),
    'Queens': df[df['neighbourhood_group_cleansed'] == 'Queens']['neighbourhood_cleansed'].unique().tolist(),
    'Bronx': df[df['neighbourhood_group_cleansed'] == 'Bronx']['neighbourhood_cleansed'].unique().tolist(),
    'Staten Island': df[df['neighbourhood_group_cleansed'] == 'Staten Island']['neighbourhood_cleansed'].unique().tolist()
    }

    # --------------------------------------------------------------------------------------

    with st.form("prediction_form"): 
        beds = st.number_input('No. of beds:', value=1)
        accom = st.number_input('No. of travellers:', value=1)
        bath = st.number_input('No. of bathrooms:', value=1)
        distrito = st.selectbox('Choose the district of New York you are interested in:', ['Choose...'] + list(districts.keys())) # First selectbox (District)
        
        # Conditioning the activation of the second selectbox
        if distrito != 'Choose...':
            barrio = st.selectbox('Select a neighborhood', list(districts.values()))


            
            #disabled_option2 = True
            #barrio = list(districts.values())
        #else:
            #disabled_option2 = False
            #neighborhood = []

        #barrio = st.selectbox('Select a neighborhood:', neighborhood, disabled=disabled_option2) # Create the second selectbox (Neighbourhood)

        # Show the selected options
        #st.write('Selected district:', distrito)
        #st.write('Selected neighborhood:', barrio)

        submit_button = st.form_submit_button(label='Predict the price')


    if submit_button:
        input_data = pd.DataFrame([[beds, accom, bath, barrio]],
                                columns=['beds', 'accommodates', 'bathrooms', 'neighbourhood_cleansed']) 

    # 1 - Encode what the user types into numbers using the mapping json.
        input_data['neighbourhood_cleansed'] = input_data['neighbourhood_cleansed'].replace(encoder)
        
    # 2 - Normalise the input data
        dtest = scaler.transform(input_data)

    # 3 - Make the prediction with the trained model
        prediction = model.predict(dtest)
        
        predicted_price = prediction[-1]  # Generally, the prediction is in the last column.
        st.write(f"### The predicted price of the accommodation is {predicted_price:.2f} €")
