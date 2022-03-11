#------------------------------------------------------------------------------------------------
###### PACKAGES
#------------------------------------------------------------------------------------------------
import pandas                   as pd
import numpy                    as np
import plotly.express           as px

import streamlit                as st

from geopy.geocoders            import Nominatim

#------------------------------------------------------------------------------------------------
###### FUNCTIONS
#------------------------------------------------------------------------------------------------

def load_mapa(df):
    data_mapa = df[['id', 'lat', 'long', 'price', 'level']].copy()

    fig = px.scatter_mapbox(data_mapa,
                           lat='lat',
                           lon='long',
                           size='price',
                           color='level',
                           color_discrete_map=True,
                           size_max = 20,
                           zoom=10)

    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(height=600, margin={'r':0, 't':0, 'l':0, 'b':0})
    #fig.show()
    return fig

def calculate_metrics(data):
# Average metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_sqft', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['Zipcode', 'Total Houses', 'Price', 'Livingroom area (SQFT)', 'Price per SQFT']

    return df

def descriptive_analysis(data):
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    min = pd.DataFrame(num_attributes.apply(np.min))
    mean = pd.DataFrame(num_attributes.apply(np.mean))
    median = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))
    max = pd.DataFrame(num_attributes.apply(np.max))

    aux = pd.concat([min, mean, median, std, max], axis=1).reset_index()
    aux.columns = ['Attributes', 'Min', 'Mean', 'Median', 'Std', 'Max']
    return aux

st.title('House Rocket Company')
st.markdown('Welcome to House Rocket Data Analysis')
st.header('Load data')

# Reading the data
@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'])
    return data

# Loading the data
original_data = get_data('dataset/kc_house_treated.csv')
data = original_data.copy()
data_by_region = original_data.copy()

# Plot map
st.title('House Rocket Map')
is_check = st.checkbox('Display Map')

# Filter bedrooms
bedrooms = st.sidebar.multiselect(
    'Number of bedrooms',
    data['bedrooms'].unique()
)

st.write('Your filter is {}'.format(bedrooms))

# Other filters
price_min = int(data['price'].min())
price_max = int(data['price'].max())
price_avg = int(data['price'].mean())

if is_check:
    price_slider = st.slider('Price Range', price_min, price_max, price_avg)
    # Select rows
    houses = data.loc[data['price'] < price_slider, ['id', 'lat', 'long','price', 'level'] ]

    # Draw map
    fig = load_mapa(houses)
    st.plotly_chart(fig)

#-------------------------------
# DATA OVERVIEW
#-------------------------------
f_attributes = st.sidebar.multiselect('Choose columns', data.columns)
f_zipcode    = st.sidebar.multiselect('Select regions by zipcode', data['zipcode'].unique())

# Filtering the data
if (f_attributes != []) & (f_zipcode != []):
    data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
elif (f_attributes == []) & (f_zipcode != []):
    data = data.loc[data['zipcode'].isin(f_zipcode)]
    data_by_region = data.loc[data['zipcode'].isin(f_zipcode)]
elif (f_attributes != []) & (f_zipcode == []):
    data = data.loc[:, f_attributes]

df1 = calculate_metrics(data_by_region)
df2 = descriptive_analysis(data)

st.title('Data overview')
c1, c2 = st.columns((1,1))

c1.markdown('Average characteristics of houses for each selected region.')
c1.dataframe(df1)
c2.markdown('Main statistics for selected columns and all selected zipcodes.')
c2.dataframe(df2)