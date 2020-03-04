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
import os

def get_raw_csv_file_path(filename):
    """
    Function that takes in the name of a CSV and returns the absolute file path
    to that file name in tanzania-water-wells/data/raw/
    """
    # Get the full absolute path to the current functions_used.py file
    current_file_path = os.path.dirname(__file__)

    # Go up two directories ("pardir" means "parent directory") to the project root
    project_root = os.path.join(current_file_path, os.pardir, os.pardir)

    # Go down two directories to data/raw
    raw_data_dir = os.path.join(project_root, 'data', 'raw')

    # Go to the CSV within data/raw
    csv_path = os.path.join(raw_data_dir, filename)

    # Return the full absolute path
    return os.path.abspath(csv_path)

def model_preprocessing(X, y, feature_list, ohe, train=True, debug=False):

    # Combine the features and target. This ensures the correct rows are dropped.
    df = pd.concat([X, y], axis=1)

    if debug: print('Beginning numerical cleaning...')
    df = numerical_clean(df, feature_list)
    if debug: print('...completed numerical cleaning.\n')
    
    if debug: print('Removing the target from the cleaned data frame...')
    target = df['status_group']
    if debug: print("---Length of target: ", len(target))
    df = df.drop(columns='status_group', axis = 1)
    if debug: print("---Shape of dataframe: ", df.shape)
    
    if debug: print("Reading the remaining columns as independent features\n")
    obj_list = obj_lister(df)
    
    if debug: print('Begining "object" cleaning...')
    ohe_df = obj_preprocessing(df, obj_list, ohe, train)
    if debug: print("---Shape of ohe_df: ", ohe_df.shape)
    if debug: print('...ending "object" cleaning.')
    
    if debug: print("Joining the cleaned numerical and object dataframes together.")
    # dropping the independent features from X
    df = df.drop(obj_list, axis=1)
    # joining the OHE dataframe to X
    model_df = df.join(ohe_df)
    if debug: print('Returning the main (independent features, X) and target (y) data frames...')
    return model_df, target


def numerical_clean(df, feature_list, debug=False):
    #this takes the df and the list of numerical features to clean
    df = df[feature_list]
    if debug: print("check: df shape = ", df.shape)
    if debug: print('---Dropping 0 longitudes...')
    df = drop_zero_long(df)
    if debug: print("check: df shape = ", df.shape)
    if debug: print("---Replace 0's with average constructor year...")
    df = con_year_avg(df)
    if debug: print("check: df shape = ", df.shape)
    if debug: print('...returning a cleaned dataframe of numerical values.\n')
    return df

def drop_zero_long(df):
    return df.drop(df[df.longitude==0].index)

def con_year_avg(df):
    con_year_nonzero = df.replace(0, np.nan)
    avg_con_years = pd.DataFrame(con_year_nonzero.groupby(['extraction_type']).mean()['construction_year'])
    df = df.join(avg_con_years, rsuffix = '_avg', on = 'extraction_type')
    df = df.reset_index()
    df = df.drop(['index'], axis = 1)
    # df['construction_year'] = wells_test.apply(con_year, axis=1)
    df = df.drop(['construction_year_avg'], axis = 1)
    return df

def obj_lister(df):
    # returns a list of columns that contain Objects
    obj_list = []
    for col in df.select_dtypes([np.object]):
        obj_list.append(col)
    return obj_list

def obj_preprocessing(df, obj_list, ohe, train = True):
    '''
    
    '''
    df_current = df[obj_list]
    # Clean the df if there are NaNs
    df = NaN_cleaning(df_current)
    #OHE data
    array_current = ohe_data(df, ohe, train)
    #return a dataframe of the OHE data
    return pd.DataFrame(array_current)


def NaN_cleaning(df, debug=False):
    # Replace NaN with "unknown" bin
    if debug: print('---Replacing NaN with "unknown" bin...')
    df = df.replace(np.nan, 'unknown')
    if debug: print(f'---Check: Number of rows with nulls: {len(df[df.isna().any(axis=1)])}...\n')
    return df.reset_index(drop=True)

def ohe_data(df, ohe, train, debug=False):
    #OHE the data
    if debug: print('Begin one hot encoding data...')
    if train:
        array_current = ohe.fit_transform(df).toarray()
    else:
        array_current = ohe.transform(df).toarray()
    if debug: print('Finish one hot encoding data...\n')
    return array_current

def calc_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
    confusion_matrix(y_test, y_pred)) 
    print('\n')
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
    print('\n')  
    print("Report : ", 
    classification_report(y_test, y_pred)) 

def plot_matrix(model,X_test,y_test):
    # 
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
