#------------------------------------------------------------------------------------------------
###### PACKAGES
#------------------------------------------------------------------------------------------------
import pandas                   as pd
import numpy                    as np
import plotly.express           as px

from geopy.geocoders            import Nominatim

#------------------------------------------------------------------------------------------------
###### FUNCTIONS
#------------------------------------------------------------------------------------------------
def show_information(df):
    print('The dataset contains {} rows and {} columns. \n'.format(data.shape[0], df.shape[1]))
    print('And its features are: \n{}'.format(df.columns))
    print('\n The number of null values for each column are: \n{}'.format(df.isna().sum()))
    print('\n The datatypes of each column are: \n{}'.format(df.dtypes))
    
    return None

def feature_engineering(df):
    # Creating the price level
    for i in range(len(df)):

        if  (df.loc[i,'price'] >= 0) & (df.loc[i,'price'] <= 321950):
            df.loc[i,'level'] = 0

        elif (df.loc[i,'price'] > 321950) & (df.loc[i,'price'] <= 450000):
            df.loc[i,'level'] = 1

        elif (df.loc[i,'price'] > 450000) & (df.loc[i,'price'] <= 645000):
            df.loc[i,'level'] = 2

        else:
            df.loc[i,'level'] = 3


    # Creating the dormitory type
    df['dormitoy_type'] = 'NA'
    for i in rage(len(df)):

        if  (df.loc[i,'bedrooms'] <= 1):
            df.loc[i,'dormitoy_type'] = 'studio'

        elif (df.loc[i,'bedrooms'] > 1) & (df.loc[i,'bedrooms'] <= 2):
            df.loc[i,'dormitoy_type'] = 'apartment'

        else:
            df.loc[i,'dormitoy_type'] = 'house'


    # Creating the price_sqft
    df.loc[:, 'price_sqft'] = df.loc[:, 'price'] / df.loc[:, 'sqft_lot']
    
    return df

def location_feature_engineering(df, user_agent='geopyExercises'):
    geolocator = Nominatim(user_agent=user_agent)
    #response = geolocator.reverse('lat, long')

    df['road'] = 'NA'
    df['house_number'] = 'NA'
    df[ 'neighbourhood'] = 'NA'
    df['county'] = 'NA'
    df['city'] = 'NA'
    df['state'] = 'NA'

    for i in range(len(df)):

        response = geolocator.reverse(df.loc[i, 'lat'].astype(str) + ',' + df.loc[i, 'long'].astype(str))
        #print('Loop {} / {}'.format(i, len(data)))

        if ('road' in response.raw['address']): 
            df.loc[i, 'road'] = response.raw['address']['road']

        elif ('house_number' in response.raw['address']):
            df.loc[i, 'house_number'] = response.raw['address']['house_number']

        elif ('neighbourhood' in response.raw['address']):
            df.loc[i, 'neighbourhood'] = response.raw['address']['neighbourhood']

        elif ('county' in response.raw['address']):
            df.loc[i, 'county'] = response.raw['address']['county']

        elif ('city' in response.raw['address']):
            df.loc[i, 'city'] = response.raw['address']['city']

        elif ('state' in response.raw['address']):
            df.loc[i, 'state'] = response.raw['address']['state']    
            
    return df
    
def load_mapa(df)
    data_mapa = df[['id', 'lat', 'long', 'price']].copy()

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
    fig.show()
    return None

def save_results(df, filename, path='dataset/'):
    filename_path = path + filename
    df.to_csv(filename_path)
    return
#------------------------------------------------------------------------------------------------
###### EXTRACTION
#------------------------------------------------------------------------------------------------
def data_extract(path, show_info=True):
    df = pd.read_csv(path)

    if show_info:
        
        show_information(df)
    
    return df

#------------------------------------------------------------------------------------------------
###### TRANSFORMATION
#------------------------------------------------------------------------------------------------
def data_transformation(df, user_agent='geopyExercises'):

    # Converting date from object to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Converting bedrooms from float to int.
    df['bedrooms'] = df['bedrooms'].astype(int)
    
    feature_engineering(df)
    
    location_feature_engineering(df, user_agent)

    return df
#------------------------------------------------------------------------------------------------
###### LOAD
#------------------------------------------------------------------------------------------------
def data_load(df, filename, path='dataset/'):
    
    save_results(df, filename, path)
    
    load_mapa(df)
    
    return None


if __name__ == '__main__'
    
    # Extract
    data = data_extract('dataset/kc_house_data.csv')
    
    # Transform
    data = data_transformation(data)
    
    # Load
    data_load(data, 'kc_house_treated')
