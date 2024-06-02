# ---------------------LIBRARIES---------------------- #
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import warnings
import base64
import os
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

df = pd.read_csv("datasets/df_short.csv")


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
                
                The listings dataset has 39881 rows and 75 columns. In this case, we have created another record with 25 columns, which we are going to work on:

                1. ``host_since``: Date the host joined Airbnb.
                2. ``host_location``: Location of the host.
                3. ``host_response_rate``: Response rate of the host.
                4. ``host_is_superhost``: Indicates whether the host is a superhost or not.
                5. ``Neighbourhood``: City zone.
                6. ``neighbourhood_cleansed``: The neighbourhood as geocoded using the latitude and longitude against neighborhoods as defined by open or public digital shapefiles.
                7. ``neighbourhood_group_cleansed``: The neighbourhood group as geocoded using the latitude and longitude against neighborhoods as defined by open or public digital shapefiles.
                8. ``latitude``: Geographical coordinates (latitude) of the property.
                9. ``longitude``: Geographical coordinates (longitude) of the property.
                10. ``property_type``: Type of property.
                11. ``room_type``: Type of room offered.
                12. ``accommodates``: Maximum number of persons that can be accommodated in the property.
                13. ``beds``: The number of bed(s).      
                14. ``price``: Price per night from
                15. ``minimum_nights``: Minimum number of night stay for the listing.
                16. ``maximum_nights``: Maximum number of night stay for the listing.
                17. ``availability_30``: Availability of the property in the next 30 days.
                18. ``availability_60``: Availability of the property in the next 60 days.
                19. ``availability_90``: Availability of the property in the next 90 days.
                20. ``availability_365``: Availability of the property in the next year.
                21.	``number_of_reviews``: Total number of reviews.
                22.	``review_scores_rating``: Overall rating.
                23. ``review_scores_location``: Location score.
                24.	``reviews_per_month``: Average number of reviews received per month for a given property.
            
                """  )      
    st.write('------')                    
    st.markdown('### **Visualisation of the pre-processed dataframe:**')
    st.dataframe(df.head(10))
    st.write('Information about the code used can be found on my GitHub: https://github.com/MarcTorrentss/airbnb_project')



# PAGE 2-------------------------------------
elif page == "Airbnb info":

    st.markdown('Here you can see the different accommodations on offer and where they are located. Zoom in on the map to see more:')
    latitud = df['latitude'].tolist()
    longitud = df['longitude'].tolist()
    coordinates = list(zip(latitud, longitud))
    # Define the initial location of the map
    latitud_1 = df['latitude'].iloc[0]
    longitud_1 = df['longitude'].iloc[0]
    # Create the Folium map with the specified starting location
    map = folium.Map(location = [latitud_1, longitud_1], zoom_start=10)
    # Adding locations to the generated Folium map
    FastMarkerCluster(data=coordinates).add_to(map) # It is used to group the closest markers into clusters.
    folium.Marker(location=[latitud_1,longitud_1]).add_to(map)
    folium_static(map)

    st.markdown("""
            ### What would you like to know? Select a tab:              
        """)

    # ---------------------TABS (pestañas)----------------------#
    tab1, tab2, tab3, tab4 = st.tabs(
        ['Accomodations','Price', 'Score','Maps']) 
    with tab1:

        ##  1. District VS No. of accommodations

        st.markdown('### District VS No. of accommodations')
        st.write('First of all, we are interested in the distribution of accommodation in each district. We can see that in Manhattan the number of accommodations or advertisements is much higher than in the other districts:')
            
        accom_district = df['neighbourhood_group_cleansed'].value_counts().sort_values(ascending=True)
            
        # Plotly bar chart
        fig = px.bar(accom_district, x=accom_district.values, y=accom_district.index, color=accom_district.values, color_continuous_scale='BrBG', text_auto = False) 
        fig.update_layout(
                title='Number of accommodations by district in New York', title_x=0.23, 
                yaxis_title='Districts',
                xaxis_title='No. of Airbnb offers',
                template='plotly_white',
                width=690, height=500, coloraxis_colorbar_title='No. of Airbnb offers')   

        st.plotly_chart(fig)


                ##  2. District VS No. of accommodations

        st.markdown('### Neighbourhood VS No. of accommodations')
        st.write('Looking closer, now we represent the distribution of airbnbs accommodation in each neighbourhood.')
            
        accom_neigh = df['neighbourhood_cleansed'].value_counts().sort_values(ascending=True)
            
        # Plotly bar chart
        fig = px.bar(accom_neigh, x=accom_neigh.values, y=accom_neigh.index, color=accom_neigh.values, color_continuous_scale='BrBG', text_auto = False) 
        fig.update_layout(
                title='Number of accommodations by neighbourhoods in New York', title_x=0.23, 
                yaxis_title='Neighbourhoods',
                xaxis_title='No. of Airbnb offers',
                template='plotly_white',
                width=690, height=500, coloraxis_colorbar_title='No. of Airbnb offers')   

        st.plotly_chart(fig)
        
        

# PAGE 3-------------------------------------
elif page == "Reviews":
    st.markdown('A **word cloud** has been created from the accommodation reviews to show you the most common words based on their size:')   

    st.markdown("<h2 style='text-align: center;'>One of the iconic cinematic images of New York City 🏢🐒</h2>", unsafe_allow_html=True)
    wordcloud = "images/nube_airbnb.png"

    st.image(wordcloud, width=500, use_column_width=True)
    st.write('-------------')
        
    st.markdown('A **sentiment analysis** of the reviews has also been carried out. You can see a visualisation of the distribution of sentiment between positive, negative or neutral:')
    
    # Open html file
    with open("images/sentimentanalysis.html", 'r', encoding='utf-8') as SentFile:
        # Read and load into source_code variable
        source_code = SentFile.read()
        print(source_code)

    # View content on Streamlit
    components.html(source_code, height=600)



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

    # Encoders
    with open("files/mapeo_barrio.json", "r") as json_file:
        # Loads the content of the JSON file into a dictionary
        encoder_neighbourhood = json.load(json_file)

    # Create a dictionary of the neighbourhoods
    districts = {
    'Brooklyn': df[df['neighbourhood_group_cleansed'] == 'Brooklyn']['neighbourhood_cleansed'].unique().tolist(),
    'Manhattan': df[df['neighbourhood_group_cleansed'] == 'Manhattan']['neighbourhood_cleansed'].unique().tolist(),
    'Queens': df[df['neighbourhood_group_cleansed'] == 'Queens']['neighbourhood_cleansed'].unique().tolist(),
    'Bronx': df[df['neighbourhood_group_cleansed'] == 'Bronx']['neighbourhood_cleansed'].unique().tolist(),
    'Staten Island': df[df['neighbourhood_group_cleansed'] == 'Staten Island']['neighbourhood_cleansed'].unique().tolist()
    }

    with open("files/mapeo_room.json", "r") as json_file:
        # Loads the content of the JSON file into a dictionary
        encoder_room = json.load(json_file)
        
    room = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
    
    # --------------------------------------------------------------------------------------

    # Initialize session state
    if 'district_selected' not in st.session_state:
        st.session_state.district_selected = False
    
    # First form, choose district
    with st.form("choose_district"):
        distrito = st.selectbox('Choose the district of New York you are interested in:', ['Choose...'] + list(districts.keys())) # First selectbox (District)
        district_button = st.form_submit_button(label='Choose district')

        if (distrito != 'Choose...' and district_button):
            st.session_state.district_selected = True


    # Second form: Prediction Form (only shown if a district has been selected)
    if st.session_state.district_selected:
    
        with st.form("prediction_form"): 
            room_type = st.selectbox('Room type:', room)
            accom = st.number_input('No. of travellers:', value=1)
            beds = st.number_input('No. of beds:', value=1)
            barrio = st.selectbox('Select a neighborhood', ['Choose...'] + districts[distrito])
            
            submit_button = st.form_submit_button(label='Predict the price')

            if submit_button: 
                input_data = pd.DataFrame([[room_type, accom, beds, barrio]], columns=['room_type', 'accommodates', 'beds', 'neighbourhood_cleansed']) 
                
                # 1 - Encode what the user types into numbers using the mapping json.
                input_data['room_type'] = input_data['room_type'].replace(encoder_room)
                input_data['neighbourhood_cleansed'] = input_data['neighbourhood_cleansed'].replace(encoder_neighbourhood)
        
                # 2 - Normalise the input data
                dtest = scaler.transform(input_data)

                # 3 - Make the prediction with the trained model
                prediction = model.predict(dtest)
        
                predicted_price = prediction[-1]  # Generally, the prediction is in the last column.
                st.write(f"### The predicted price of the accommodation is {predicted_price:.2f} €")
