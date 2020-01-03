import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

def model_preprocessing(df, feature_list, ohe, train=True):
    print('Beginning numerical cleaning...')
    df = numerical_clean(df, feature_list)
    print('Completed numerical cleaning.\n')
    
    print('Removing the target from the cleaned data frame...')
    target = df['status_group']
    print("---Length of target: ", len(target))
    df = df.drop(columns='status_group', axis = 1)
    print("---Shape of dataframe: ", df.shape)
    
    print("Reading the remaining columns as independent features\n")
    obj_list = obj_lister(df)
    
    print('Begining "object" cleaning...')
    ohe_df = obj_preprocessing(df, obj_list, ohe, train)
    print("---Shape of ohe_df: ", ohe_df.shape)
    print('...ending "object" cleaning.')
    
    print("Joining the cleaned numerical and object dataframes together.")
    # dropping the independent features from X
    df = df.drop(obj_list, axis=1)
    # joining the OHE dataframe to X
    model_df = df.join(ohe_df)
    print('Returning the main (independent features, X) and target (y) data frames...')
    return model_df, target


def numerical_clean(df, feature_list):
    #this takes the df and the list of numerical features to clean
    df = df[feature_list]
    print("check: df shape = ", df.shape)
    print('---Dropping 0 longitudes...')
    df = drop_zero_long(df)
    print("check: df shape = ", df.shape)
    print("---Replace 0's with average constructor year...")
    df = con_year_avg(df)
    print("check: df shape = ", df.shape)
    print('...returning a cleaned dataframe of numerical values.')
    return df

def drop_zero_long(df):
    return df.drop(df[df.longitude==0].index)

def con_year_avg(df):
    con_year_nonzero = df.replace(0, np.nan)
    avg_con_years = pd.DataFrame(con_year_nonzero.groupby(['extraction_type']).mean()['construction_year'])
    df = df.join(avg_con_years, rsuffix = '_avg', on = 'extraction_type')
    df = df.reset_index()
    df = df.drop(['index'], axis = 1)
    # df['construction_year'] = df.apply(con_year, axis=1)
    df = df.drop(['construction_year_avg'], axis = 1)
    return df

def con_year(row: pd.DataFrame) -> int:
        ## This function replaces 
        if (row['construction_year'] == 0):
            return int(row['construction_year_avg'])
        else:  
            return int(row['construction_year'])


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


def NaN_cleaning(df):
    import numpy
    # Replace NaN with "unknown" bin
    print('---Replacing NaN with "unknown" bin...')
    df = df.replace(numpy.nan, 'unknown')
    print(f'---Check: Number of rows with nulls: {len(df[df.isna().any(axis=1)])}...\n')
    return df.reset_index(drop=True)

def ohe_data(df, ohe, train):
    #OHE the data
    print('Begin one hot encoding data...')
    if train:
        array_current = ohe.fit_transform(df).toarray()
    else:
        array_current = ohe.transform(df).toarray()
    print('Finish one hot encoding data...\n')
    return array_current


# Function to calculate accuracy 
# from: https://www.geeksforgeeks.org/decision-tree-implementation-python/
def calc_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
    confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 