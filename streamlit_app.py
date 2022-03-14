#------------------------------------------------------------------------------------------------
###### PACKAGES
#------------------------------------------------------------------------------------------------
from time import strptime
import pandas                   as pd
import numpy                    as np
import plotly.express           as px
import datetime                 as dt

import streamlit                as st

import geopandas
import folium

from streamlit_folium           import folium_static
from folium.plugins             import MarkerCluster
from geopy.geocoders            import Nominatim



#------------------------------------------------------------------------------------------------
###### FUNCTIONS
#------------------------------------------------------------------------------------------------

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'])
    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)

    return geofile

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

st.set_page_config( layout='wide' )
st.title('House Rocket Company')
st.markdown('Welcome to House Rocket Data Analysis')
st.header('Load data')

# Loading the data
original_data = get_data('dataset/kc_house_treated.csv')
data = original_data.copy()
data_by_region = original_data.copy()

# Get the profile
url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
geofile = get_geofile(url)

# Plot map
st.title('House Rocket Map')
is_check = st.checkbox('Display Map')

# Filter bedrooms
bedrooms = st.sidebar.multiselect(
    'Number of bedrooms',
    sorted(set(data['bedrooms'].unique()))
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
    st.plotly_chart(fig, use_container_width=True)

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

#-------------------------------
# PORTFOLIO DENSITY
#-------------------------------

st.title('Region Overview')

c1, c2 = st.columns((1,1))

c1.header( 'Portfolio Density' )
# Building basic map
density_map = folium.Map( location=[data['lat'].mean(), data['long'].mean() ], default_zoom_start=15 ) 

marker_cluster1 = MarkerCluster().add_to( density_map )

for name, row in data.iterrows():
    folium.Marker( [row['lat'], row['long'] ], 
        popup='Sold ${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format( row['price'],
                                     row['date'],
                                     row['sqft_living'],
                                     row['bedrooms'],
                                     row['bathrooms'],
                                     row['yr_built'] ) ).add_to( marker_cluster1 )

with c1:
    folium_static( density_map )


# Region Price Map
c2.header( 'Price Density' )

df = data_by_region[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
df.columns = ['ZIP', 'PRICE']
geofile = geofile[geofile['ZIP'].isin( df['ZIP'].tolist() )]

region_price_map = folium.Map( location=[data['lat'].mean(), 
                               data['long'].mean() ],
                               default_zoom_start=15 )

#marker_cluster2 = MarkerCluster().add_to( density_map )

region_price_map.choropleth(data = df,
                            geo_data = geofile,
                            columns=['ZIP', 'PRICE'],
                            key_on='feature.properties.ZIP',
                            fill_color='YlOrRd',
                            fill_opacity = 0.7,
                            line_opacity = 0.2,
                            legend_name='Average Price' )

with c2:
    folium_static( region_price_map )

#-------------------------------
# COMMERCIAL OVERVIEW
#-------------------------------

st.title('Commercial attributes')
st.sidebar.title('Commercial Options')

# Deciding whether show data by region
is_region_data = st.sidebar.checkbox('Filter houses on selected zipcodes')

# Filter for year built
if is_region_data:
    commercial_data = data_by_region.copy()
else:
    commercial_data = data.copy()

# Year built filter
st.sidebar.subheader('Select Max Year built')
min_year_built = int(commercial_data['yr_built'].min())
max_year_built = int(commercial_data['yr_built'].max())
f_yr_built = st.sidebar.slider('Year built', min_year_built, max_year_built, max_year_built)

# Date filter
st.sidebar.subheader('Select Max Date')
commercial_data['date'] = pd.to_datetime(commercial_data['date']).apply(lambda x: x.date())
min_date = dt.datetime.strptime(str(commercial_data['date'].min()), '%Y-%m-%d')
max_date = dt.datetime.strptime(str(commercial_data['date'].max()), '%Y-%m-%d')
f_date = st.sidebar.slider('Sold Date', min_date, max_date, max_date)

# Constructing year built graph
df3 = commercial_data[['date', 'price', 'price_sqft', 'yr_built']].groupby('yr_built').mean().reset_index()
df3 = df3.loc[df3['yr_built'] <= f_yr_built]
fig = px.line(df3, x='yr_built', y='price')
st.plotly_chart(fig, use_container_width=True)

# Constructing average price per day graph
df4 = commercial_data[['date', 'price', 'price_sqft', 'yr_built']].groupby('date').mean().reset_index()
df4 = df4.loc[df4['date'] <= dt.date(f_date.year,f_date.month, f_date.day)]
fig = px.line(df4, x='date', y='price')
st.plotly_chart(fig, use_container_width=True)

#-------------------------------
# HISTOGRAMS
#-------------------------------
st.sidebar.subheader('Select Max Price')

# Filtering max price
min_price = int(commercial_data['price'].min())
max_price = int(commercial_data['price'].max())
avg_price = int(commercial_data['price'].mean())
f_price = st.sidebar.slider('House Price', min_price, max_price, avg_price)

# Other filters
f_bedrooms = st.sidebar.selectbox('Max number of bedrooms in Price range', sorted(set(commercial_data['bedrooms'].unique())))
f_bathrooms = st.sidebar.selectbox('Max number of bathrooms in Price range', sorted(set(commercial_data['bathrooms'].unique())))
f_floors = st.sidebar.selectbox('Max number of floors in Price range', sorted(set(commercial_data['floors'].unique())))
f_waterview = st.sidebar.checkbox('Only houses with waterview')


st.header('Price Distribution')
fig = px.histogram(commercial_data, x='price', nbins=50)
st.plotly_chart(fig, use_container_width=True)

# Price data plot
commercial_data = commercial_data.loc[commercial_data['price'] <= f_price]
fig = px.histogram(commercial_data, x='price', nbins=50)
st.plotly_chart(fig, use_container_width=True)


st.header('Main Characteristics Distribution')
c1, c2 = st.columns((1,1))

# Houses per bedrooms
c1.header('Houses per number of bedrooms')
houses1 = commercial_data[commercial_data['bedrooms'] <= f_bedrooms]
fig = px.histogram(houses1, x='bedrooms', nbins=10)
c1.plotly_chart(fig, use_container_width=True)

# Houses per bedrooms
c1.header('House per number of floors')
houses3 = commercial_data[commercial_data['floors'] <= f_floors]
fig = px.histogram(houses3, x='floors', nbins=10)
c1.plotly_chart(fig, use_container_width=True)

# Houses per bathrooms 
c2.header('Houses per number of bathrooms')
houses2 = commercial_data[commercial_data['bathrooms'] <= f_bathrooms]
fig = px.histogram(houses2, x='bathrooms', nbins=10)
c2.plotly_chart(fig, use_container_width=True)

# Waterfront
if f_waterview:
    houses4 = commercial_data[commercial_data['waterfront'] == 1]
else:
    houses4 = commercial_data.copy()
c2.header('Houses per waterfront')
fig = px.histogram(houses4, x='waterfront', nbins=5)
c2.plotly_chart(fig, use_container_width=True)
