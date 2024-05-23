<div align="center">
  
# Airbnb's Impact on New York's Housing Market 🏛️

</div>

<p align="center">
  <img src='https://static.anuevayork.com/wp-content/uploads/2020/06/16084125/Guia-de-Nueva-York-en-espanol-Empire-State-Building-scaled.jpg' width="1000" height="600" />
</p>

This project will explore a dataset of housing advertised on the Airbnb platform in the city of New York, the city that never sleeps (never said better jeje). In this way, it will analyse the impact of Airbnb on the housing market in this very touristic city, rental patterns, prices, tourist flows and safety in the city.

This analysis will allow us to understand the impact of this platform and to formulate possible regulations to address the associated problems.

The data to be worked with corresponds to the register of Airbnb listings published in the city of Rome as of 7 September 2022.

### Project characteristics

- **Data**: Data for this project was obtained from the website [Inside Airbnb](https://insideairbnb.com/get-the-data/).
- **Code**: The code used is located in the folder ``notebooks``, and includes three jupyter notebooks with different sections:
  
    1. ``1_Preprocessing_EDA.ipynb``:
        - Loading the libraries and reading the different datasets.
        - Datasets information.
        - Data pre-processing: null and outlier repair, fixing values, select valuable information...
        - Exploratory data analysis (EDA), including visualisation of interactive maps and other graphics.
        
    2. ``2_NLP.ipynb``. **Natural Language Processing**:
         - Creation of a **word cloud** visualisation ilustrating the frequency and importance of words in the textual data, based on the list reviews.
         - **Sentiment analysis** of reviews and distribution of sentiment between positive, negative or neutral. Visualization.
           
    4. ``3_ML_pricepredictor.ipynb``. Implementation of **machine learning models** to predict accommodation prices (regression model):
        - Preprocessing filtered dataframe used to train models.
        - Data splitting using train_test_split() from scikit-learn.
        - Data normalisation using ``StandardScaler()``.
        - Training of different regression models.

- **Streamlit application**: An interactive Streamlit application has been developed that allows exploration and visualisation of the analysed data. It is available at https://airbnb-ny.streamlit.app/

### Running instructions 💻

To run this project on your local machine, follow the steps below:

1. Clone this repository onto your local machine.
2. Install the necessary dependencies by running ``pip install -r requirements.txt``.
3. Run ``app_airbnb.py`` and make sure you have downloaded the ``outputs``, ``img``,``html`` and ``models`` folders in the same environment. Next, open a terminal in the app directory and run the following command ``streamlit run app_airbnb.py``.
4. This will open a web browser ``http://localhost:8501/`` which will take you to the application.

## Streamlit App demo 📹

### To do ⚙️

- [x] Implementation of a machine learning model to predict accommodation prices based on a set of input parameters. 
- [x] Sentiment Analysis using NLTK library

## Contact 📧
If you have any questions or suggestions about this project, please feel free to contact me. You can get in touch with me through my social media channels.
