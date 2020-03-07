# Be sure to train/test split before processing DFs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from ipyleaflet import Map, basemaps, basemap_to_tiles, CircleMarker, LayerGroup


def model_preprocessing(df, feature_list, ohe, train=True):
    """
    This function takes a dataframe, list of desired attributes and an instance of a 
    one hot encoder and returns:
    1. the target
    2. a joined dataframe with the OHE'd data and a datafram that has been cleaned
    """
    # print('Beginning numerical cleaning...')
    df = numerical_clean(df, feature_list)
    # print('...completed numerical cleaning.\n')
    
    # print('Removing the target from the cleaned data frame...')
    target = df['status_group']
    # print("---Length of target: ", len(target))
    df = df.drop(columns='status_group', axis = 1)
    # print("---Shape of dataframe: ", df.shape)
    
    # print("Reading the remaining columns as independent features\n")
    obj_list = obj_lister(df)
    
    # print('Begining "object" cleaning...')
    ohe_df = obj_preprocessing(df, obj_list, ohe, train)
    # print("---Shape of ohe_df: ", ohe_df.shape)
    # print('...ending "object" cleaning.')
    
    # print("Joining the cleaned numerical and object dataframes together.")
    # dropping the independent features from X
    df = df.drop(obj_list, axis=1)
    # joining the OHE dataframe to X
    model_df = df.join(ohe_df)
    # print('Returning the main (independent features, X) and target (y) data frames...')
    return model_df, target


def numerical_clean(df, feature_list):
    '''
    This function is part of the cleaning stage. It drops rows with longitude = 0.
    '''
    #this takes the df and the list of numerical features to clean
    df = df[feature_list]
    # print("check: df shape = ", df.shape)
    # print('---Dropping 0 longitudes...')
    df = drop_zero_long(df)
    # print("check: df shape = ", df.shape)
    # print("---Replace 0's with average constructor year...")
    df = con_year_avg(df)
    # print("check: df shape = ", df.shape)
    # print('...returning a cleaned dataframe of numerical values.\n')
    return df

def drop_zero_long(df):
    '''
    This function is part of the cleaning stage. It drops rows with 0 longitude.
    '''
    return df.drop(df[df.longitude==0].index)

def con_year_avg(df):
    """
    This function takes in a daftaframe and returns a dataframe that
    has imputed feature (construction year) deremined by the average of construction type
    """
    con_year_nonzero = df.replace(0, np.nan)
    avg_con_years = pd.DataFrame(con_year_nonzero.groupby(['extraction_type']).mean()['construction_year'])
    df = df.join(avg_con_years, rsuffix = '_avg', on = 'extraction_type')
    df = df.reset_index()
    df = df.drop(['index'], axis = 1)
    # df['construction_year'] = wells_test.apply(con_year, axis=1)
    df = df.drop(['construction_year_avg'], axis = 1)
    return df

def obj_lister(df):
    """
    This function take in a dataframe and returns a list of columns that contain objects
    """
    obj_list = []
    for col in df.select_dtypes([np.object]):
        obj_list.append(col)
    return obj_list

def obj_preprocessing(df, obj_list, ohe, train = True):
    """
    This function takes a dataframe, list of columns as objects and an instance of a 
    one hot encoder and returns a dataframe that has been OHE. It is the driver code for
    categorical data processing. Steps:
    1. Clean the df if there are NaNs
    2. OHE's data
    """
    df_current = df[obj_list]
    # Clean the df if there are NaNs
    df = NaN_cleaning(df_current)
    #OHE data
    array_current = ohe_data(df, ohe, train)
    #return a dataframe of the OHE data
    return pd.DataFrame(array_current)


def NaN_cleaning(df):
    """
    This function thakes in a dataframe and cleans nulls by replacing with text. It returns a
    cleaned dataframe.
    """
    # Replace NaN with "unknown" bin
    # print('---Replacing NaN with "unknown" bin...')
    df = df.replace(np.nan, 'unknown')
    # print(f'---Check: Number of rows with nulls: {len(df[df.isna().any(axis=1)])}...\n')
    return df.reset_index(drop=True)

def ohe_data(df, ohe, train):
    """
    This function takes in a dataframe, OHE instance, and training data.
    It OHE's the  categorical data and returns an array.
    """
    # print('Begin one hot encoding data...')
    if train:
        array_current = ohe.fit_transform(df).toarray()
    else:
        array_current = ohe.transform(df).toarray()
    # print('Finish one hot encoding data...\n')
    return array_current

def calc_accuracy(y_test, y_pred): 
    """
    This function takes in test and predition values. It prints:
    1. a confusion matrix
    2. the accuracy score
    3. the classification report
    It doesn't return any values.
    """
    print("Confusion Matrix: ", 
    confusion_matrix(y_test, y_pred)) 
    print('\n')
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
    print('\n')  
    print("Report : ", 
    classification_report(y_test, y_pred)) 

def plot_matrix(model, X_test, y_test):
    """
    This function takes in a model and test values and creates a plot of the confusion matrix.
    """
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, X_test, y_test,
                                display_labels = ['Functional', 'Needs Repair'],
                                cmap=plt.cm.Blues,
                                normalize = normalize)
        disp.ax_.set_title(title)
#     print(title)
#     print(disp.confusion_matrix)
    
    plt.show()

def make_map(X_test):
    """
    This function takes in test vales and returns a map of the locations of test wells
    """
    m = Map(center=(-6, 35),
            zoom=5, 
            scroll_wheel_zoom=True)
    def create_marker(row):
        lat_lon = (row["latitude"], row["longitude"])
        return CircleMarker(location=lat_lon,
                        draggable=False,
                        fill_color="#055a8c",
                        fill_opacity=0.35,
                        radius=1,
                        stroke=False)
    markers = X_test.apply(create_marker, axis=1)
    layer_group = LayerGroup(layers=tuple(markers.values))
    m.add_layer(layer_group)
    return m