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
from streamlit_folium import folium_static, st_folium
from branca.colormap import LinearColormap
import geopandas as gpd
from shapely.geometry import Point

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ['Accomodations', 'Room and Property', 'Prices', 'Safety', 'Scores'])
    
    with tab1:

        ##  1. District VS No. of accommodations

        st.markdown('### District VS No. of accommodations')
        st.write('First of all, we are interested in the distribution of accommodation in each district. We can see that in ``Manhattan`` and ``Brooklyn`` the number of accommodations or advertisements is much higher than in the other districts:')
            
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
        accom_neigh = accom_neigh[accom_neigh > 500]
        
        # Plotly bar chart
        fig = px.bar(accom_neigh, x=accom_neigh.values, y=accom_neigh.index, color=accom_neigh.values, color_continuous_scale='BrBG', text_auto = False) 
        fig.update_layout(
                title='Number of accommodations by neighbourhoods in New York', title_x=0.23, 
                yaxis_title='Neighbourhoods',
                xaxis_title='No. of Airbnb offers',
                template='plotly_white',
                width=690, height=500, coloraxis_colorbar_title='No. of Airbnb offers')   

        st.plotly_chart(fig)


    with tab2:

        ##  3. Room types

        st.markdown('### Types of room bookings')
        st.write('We see that almost all Airbnbs in the city are ``entire apartments`` and ``private rooms``.')
            
        accom_rooms = df['room_type'].value_counts().sort_values(ascending=True)
            
        # Plotly bar chart
        fig = px.bar(accom_rooms, x=accom_rooms.values, y=accom_rooms.index, color=accom_rooms.values, text_auto = False) 
        fig.update_layout(
                title='Room types in New York', title_x=0.23, 
                yaxis_title='Room types',
                xaxis_title='No. of Airbnb offers',
                template='plotly_white',
                width=690, height=500, coloraxis_colorbar_title='No. of Airbnb offers')   

        st.plotly_chart(fig)


        ##  4. Property types

        freq_proptype = df['property_type'].value_counts()
        freq_proptype = freq_proptype[freq_proptype > 45]

        # We modify the property type values
        list_proptype = freq_proptype.index.to_list()
        dict_proptype = {list_proptype[0]:'Apartment',
        list_proptype[1]:'Apartment',
        list_proptype[2]:'House',
        list_proptype[3]:'Condominium',
        list_proptype[4]:'House',
        list_proptype[5]:'Townhouse',
        list_proptype[6]:'Loft',
        list_proptype[7]:'Hotel',
        list_proptype[8]:'Townhouse',
        list_proptype[9]:'Apartment',
        list_proptype[10]:'Condominium',
        list_proptype[11]:'Hotel',
        list_proptype[12]:'Apartment',
        list_proptype[13]:'Guest suite',
        list_proptype[14]:'Apartment',
        list_proptype[15]:'Loft',
        list_proptype[16]:'Guest suite',
        list_proptype[17]:'House',
        list_proptype[18]:'Apartment',
        list_proptype[19]:'House',
        list_proptype[20]:'Aparthotel',
        list_proptype[21]:'B&B',
        list_proptype[22]:'Apartment',
        list_proptype[23]:'House',
        list_proptype[24]:'House',}
        df = df.replace({"property_type": dict_proptype})
        
        st.markdown('### Types of properties')
        st.write('We can see that the vast majority of Airbnbs in New York are ``apartments``, with a significant difference compared to the second category, which is ``houses``. This makes sense since when you see the city buildings.')

        accom_properties = df['property_type'].value_counts().sort_values(ascending=True)
        accom_properties = accom_properties[accom_properties > 100]
        
        # Plotly bar chart
        fig = px.bar(accom_properties, x=accom_properties.values, y=accom_properties.index, color=accom_properties.values, text_auto = False) 
        fig.update_layout(
                title='Property types in New York', title_x=0.23, 
                yaxis_title='Property types',
                xaxis_title='No. of Airbnb offers',
                template='plotly_white',
                width=690, height=500, coloraxis_colorbar_title='No. of Airbnb offers')   

        st.plotly_chart(fig)


        ##  6. Room types

        st.markdown('### Number of listings by accomodates')
        st.write('The majority of bookings are for ``1 to 4 people``, with ``2 people`` standing out with a significant advantage.')
            
        accom_accomodates = df['accommodates'].value_counts().sort_values(ascending=True)
            
        # Plotly bar chart
        fig = px.bar(accom_accomodates, x=accom_accomodates.index, y=accom_accomodates.values, text_auto = False) 
        fig.update_layout(
                title='Number of listings by accomodates in New York', title_x=0.23, 
                yaxis_title='No. of Airbnb offers',
                xaxis_title='No. of accommodates',
                template='plotly_white',
                width=690, height=500, coloraxis_colorbar_title='No. of Airbnb offers')   

        st.plotly_chart(fig)
    

    with tab3:
                
        ## 7. Neighbourhood VS Average price

        st.markdown('### Neighbourhood VS Average price')
        st.write('It would also be interesting to know the average price for each neighbourhood')
            
        adam = gpd.read_file("files/neighbourhoods.geojson")

        with st.form("choose_accomodates"):
            No_accomodates = st.selectbox('Choose the number of accomodates:', ['Choose...'] + list(range(1, 17))) # Selectbox (Accomodates)
            accomodates_button = st.form_submit_button(label='Choose No. of accomodates')
        
            if accomodates_button:
            
                select_people = df[df['accommodates'] == No_accomodates]
        
                feq1 = select_people.groupby(['neighbourhood_cleansed'])['price'].mean()
                feq1 = feq1.sort_values(ascending=False)
                feq1 = feq1.to_frame().reset_index()
                feq1 = feq1.rename(columns = {"neighbourhood_cleansed":"neighbourhood", "price":"average_price"})
                adam = pd.merge(adam, feq1, on='neighbourhood', how='left')
                adam.rename(columns={'price': 'average_price'}, inplace=True)
                adam.average_price = adam.average_price.round(decimals=0)
                adam = adam.dropna()
        
                map_dict = adam.set_index('neighbourhood')['average_price'].to_dict()
                color_scale = LinearColormap(['green','yellow','orange','red','brown'], vmin = min(map_dict.values()), vmax = max(map_dict.values()))

                def get_color(feature):
                    value = map_dict.get(feature['properties']['neighbourhood'])
                    return color_scale(value)
            
                # Create the Folium map with the specified starting location
                map2 = folium.Map(location = [latitud_1, longitud_1], zoom_start=10)
                folium.GeoJson(data=adam, name='New york',
                    tooltip=folium.features.GeoJsonTooltip(fields=['neighbourhood', 'average_price'],
                                                labels=True,
                                                sticky=False),

                    style_function= lambda feature: {
                        'fillColor': get_color(feature),
                        'color': 'black',
                        'weight': 1,
                        'dashArray': '5, 5',
                        'fillOpacity':0.5
                        },
                    highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}).add_to(map2)      
            
                folium_static(map2)
                st.write('We see that the highest concentration of the highest average daily prices for Airbnb is in tourist zone, ``Manhattan`` and in the ``Brooklyn`` area that is close to the center.')


    with tab4:
                
        ## 8. Neighbourhood VS Safety

        st.markdown('### Neighbourhood VS Safety')
        st.write('It would also be interesting to know the amount of felonies, misdemeanors, and violations reported to the New York City Police Department for each neighbourhood')
            
        safe = pd.read_csv("datasets/safety_limpio.csv")

        # We create an interactive map of the crime count by neighborhood
        lats_s = safe['Latitude'].tolist()
        lons_s = safe['Longitude'].tolist()
        locations_s = list(zip(lats_s, lons_s))
        # Create the Folium map with the specified starting location
        map3 = folium.Map(location = [latitud_1, longitud_1], zoom_start=10)
        # Adding locations to the generated Folium map
        FastMarkerCluster(data=locations_s).add_to(map3) # It is used to group the closest markers into clusters.
        folium.Marker(location=[latitud_1,longitud_1]).add_to(map3)
        folium_static(map3)

        st.write('If we talk about reports, the majority have been informed in the tourist zone.')
    

    with tab5:
                
        ## 9. Neighbourhood VS Scores

        st.markdown('### Neighbourhood VS Scores')
        st.write('It would also be interesting to know the the neighborhoods with their respective airbnb average ratings')

        adam2 = gpd.read_file("files/neighbourhoods.geojson")
        
        feq2 = df.groupby('neighbourhood_cleansed')['review_scores_location'].mean().sort_values(ascending=True)
        feq2 = feq2.to_frame().reset_index()
        feq2 = feq2.rename(columns = {"neighbourhood_cleansed":"neighbourhood", "review_scores_location":"average_review"})
        adam2 = pd.merge(adam2, feq2, on='neighbourhood', how='left')
        adam2 = adam2.dropna()
        
        map_dict2 = adam2.set_index('neighbourhood')['average_price'].to_dict()
        color_scale2 = LinearColormap(['green','yellow','orange','red','brown'], vmin = min(map_dict2.values()), vmax = max(map_dict2.values()))

        def get_color(feature):
            value = map_dict2.get(feature['properties']['neighbourhood'])
            return color_scale2(value)
            
        # Create the Folium map with the specified starting location
        map3 = folium.Map(location = [latitud_1, longitud_1], zoom_start=10)
        folium.GeoJson(data=adam2, name='New york',
            tooltip=folium.features.GeoJsonTooltip(fields=['neighbourhood', 'average_price'],
                                        labels=True,
                                        sticky=False),

            style_function= lambda feature: {
                'fillColor': get_color(feature),
                'color': 'black',
                'weight': 1,
                'dashArray': '5, 5',
                'fillOpacity':0.5
                },
            highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}).add_to(map3)      
            
        folium_static(map3)



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
