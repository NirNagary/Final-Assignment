
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:12:03 2023

@author: annak
"""
import re
import csv
import pandas as pd
import numpy as np 
from datetime import timedelta
from datetime import datetime
import datetime as dt
import unicodedata
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import FunctionTransformer
import pickle

### נתונים 
data=pd.read_excel(r"C:\Users\Nir\Desktop\ספרים וחוברות\שנה 3\כרייה וניתוח נתונים\output_all_students_Train_v10.xlsx")


def prepare_data(data):
    data.dropna(subset=['price'], inplace=True)

    ##### פונקציות לטיפול בנתונים

    def convert(txt):
        if not isinstance(txt, str):
            return txt
        
        txt = txt.replace(",", "")
        txt = txt.replace("'", "")
        
        return txt

    def extract_numbers(txt):
        if not isinstance(txt, (str, bytes, int, float)):
            return ''

        pattern = r'\b\d+(\.\d+)?\b'
        match = re.search(pattern, str(txt))
        
        if match:
            return match.group()
        else:
            return ''

    def convert_to_float(txt):
        try:
            return float(txt)
        except ValueError:
            return None




    ## יש ריקים צריך לחשוב מה לעשות איתם
    ## 8 עמודות טקסטואליים שהגיוני להוריד הכל חוץ מהא-ב


    def extract_letters(txt):
        if not isinstance(txt, (str, bytes)):
            return ''
        
        pattern = r'[\[\]\w]+'
        letters = re.findall(pattern, txt)
        return ' '.join(letters)

    def extract_floor(txt):
        if not isinstance(txt, (str, bytes)):
            return ''
        pattern = r'\d+'
        floor = re.search(pattern, txt)
        if floor:
            return int(floor.group())
        else:
            return None
        
        
    def extract_total_floors(txt):
        if not isinstance(txt, (str, bytes)):
            return ''
        pattern = r'מתוך (\d+)'
        total_floors = re.search(pattern, txt)
        if total_floors:
            return int(total_floors.group(1))
        else:
            return None
       
        
       
    def categorize_date(date):
        if isinstance(date, str):
            return date  # Return string value as it is
        elif pd.isnull(date):
            return "Missing Value"  # Categorize missing values (NaT)
        else:
            current_date = datetime.now().date()

            if pd.isnull(date.date()):
                return "Missing Value"  # Categorize missing values in datetime objects

            delta = current_date - date.date()

            if delta > pd.Timedelta(days=365):
                return "More than a year"
            elif delta > pd.Timedelta(days=182):
                return "More than half a year"
            else:
                return "Less than half a year"






    def transform_value(value):
        if value == 'גמיש':
            return 'flexible'
        
        elif value == 'מיידי':
            return 'immediately'
        
        elif unicodedata.normalize('NFKC', str(value).strip()) == "גמיש":
            return 'flexible'
        
       
        elif value == 'לא צויין':
            return 'not defined'
        
        else:
            return value

    def transform_to_binary(df, columns):
        value_mapping = {
            True: 1,
            False: 0,
            'yes': 1,
            'no': 0,
        }

        regex_pattern = re.compile(r'\b(?:יש(?:\s+[^a-zA-Z0-9]+[^a-zA-Z0-9\s]+)*)|(?:כן(?:\s+[^a-zA-Z0-9]+[^a-zA-Z0-9\s]+)*)\b', re.UNICODE)

        for column in columns:
            df[column] = df[column].replace(value_mapping).fillna(df[column].astype(str).apply(lambda x: 1 if re.findall(regex_pattern, x) else 0))

        return df






    def convert_hebrew_words(series):
        pattern_yes = re.compile(r'(כן|יש)')
        pattern_no = re.compile(r'(לא|אין)')

        def convert_word(word):
            if isinstance(word, str):
                if re.search(pattern_yes, word):
                    return 1
                elif re.search(pattern_no, word):
                    return 0
            return word

        return series.apply(convert_word)




    def Drop_na_column(df,column):
        return df.dropna(subset=[column],inplace=True)
        




    ###################### כל השינויים 
        
    data['price'] = data['price'].apply(convert)
    data['price'] = data['price'].apply(extract_numbers)
    data['price']=data['price'].apply(convert_to_float)
    data.dropna(subset=['price'], inplace=True)
    data['Area']=data['Area'].apply(convert)
    data['Area'] = data['Area'].apply(extract_numbers)
    data['City']=data['City'].apply(extract_letters)
    data['type']=data['type'].apply(extract_letters)
    data['Street']=data['Street'].apply(extract_letters)
    data['city_area']=data['city_area'].apply(extract_letters)
    data['condition ']=data['condition '].apply(extract_letters)
    data['description ']=data['description '].apply(extract_letters)
    data['floor']=data['floor_out_of'].apply(extract_floor)
    data['total_floors']=data['floor_out_of'].apply(extract_total_floors)
    data['entrance_date'] = data['entranceDate '].apply(transform_value)
    data['entrance_date'] = data['entrance_date'].apply(categorize_date)
    data['hasElevator ']=convert_hebrew_words(data['hasElevator '])
    data['hasParking ']=convert_hebrew_words(data['hasParking '])
    data['hasBars ']=convert_hebrew_words(data['hasBars '])
    data['hasStorage ']=convert_hebrew_words(data['hasStorage '])
    data['hasAirCondition ']=convert_hebrew_words(data['hasAirCondition '])
    data['hasBalcony ']=convert_hebrew_words(data['hasBalcony '])
    data['hasMamad ']=convert_hebrew_words(data['hasMamad '])
    data['handicapFriendly ']=convert_hebrew_words(data['handicapFriendly '])
    data=transform_to_binary(data,['hasElevator ','hasParking ','hasBars ','hasStorage ','hasAirCondition ','hasBalcony ','hasMamad ','handicapFriendly '])
    #data['room_number']=data['room_number'].apply(convert)
    data['room_number'] = data['room_number'].apply(extract_numbers)

    data['num_of_images'] = data['num_of_images'].apply(extract_numbers)

    columns_to_delete = ['floor_out_of ','number_in_street','entranceDate ']
    data = data.drop(columns=columns_to_delete, errors='ignore')
    data = data.drop(['publishedDays ','floor_out_of'],axis=1)

    for column in ['room_number','Area','num_of_images','hasElevator ','hasParking ','hasBars ','hasStorage ','hasAirCondition ','hasBalcony ','hasMamad ','handicapFriendly ','floor','total_floors']:
        data[column] = pd.to_numeric(data[column], errors='coerce')


    # =============================================================================
    # =============================================================================
    #  
    # data.replace(['',' '],np.nan,inplace=True)
    # data.dropna(subset=['Area'],inplace=True) ## area iscursal parameter for the price prediction
    # 
    # ## changing values from nan to 0 0 in floor to houses and not apartments
    condition  = (data['type'].isin(['בית פרטי', 'דו משפחתי', 'קוטג', 'קוטג טורי','בניין','מגרש'])) & (data['floor'].isnull() | data['total_floors'].isnull())
    data.loc[condition, ['floor', 'total_floors']] = 0
    # 
    # ## change nan in floor where the apartment is with garden.
    # condition2  = (data['type'] == 'דירת גן') & (data['floor'].isnull() | data['total_floors'].isnull())
    # data.loc[condition2,['floor']] = 1
    # 
    data['city_area'] = data['city_area'].fillna('אין')
    
    return data




data=prepare_data(data)






####  splitting the data ## 
X = data.drop('price',axis = 1)
X = X.drop(['description ','num_of_images','handicapFriendly ','hasBars ','furniture ','entrance_date'],axis =1)
#X = X.drop(['description ','city_area'],axis =1)
y = data.price






cat_cols = ['City','type','city_area','Street','condition ']  # List of categorical column names
#num_cols = ['room_number','Area','num_of_images','hasElevator ','hasParking ','hasBars ','hasStorage ','hasAirCondition ','hasBalcony ','hasMamad ','handicapFriendly ','floor','total_floors']  # List of numerical column names
#cat_cols = ['City','type','Street','condition ','furniture ','entrance_date']  # List of categorical column names
num_cols = ['room_number','Area','hasElevator ','hasParking ','hasStorage ','hasAirCondition ','hasBalcony ','hasMamad ','floor','total_floors']  # List of numerical column names



# Feature Engineering
log_transformer = FunctionTransformer(np.log1p, validate=True)
interaction_transformer = PolynomialFeatures(degree=2, interaction_only=True)

numerical_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='median', add_indicator=False)),
    ('log_transform', log_transformer),
    ('scaling', RobustScaler())
])

poly_pipeline = Pipeline([
    ('polynomial_features', interaction_transformer),
    ('scaling', RobustScaler())
])

categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='constant', add_indicator=False, fill_value='missing')),
    ('one_hot_encoding', OneHotEncoder( handle_unknown='ignore'))
])





column_transformer = ColumnTransformer([
    ('numerical_pipeline', numerical_pipeline, num_cols),
    ('categorical_preprocessing', categorical_pipeline, cat_cols),
], remainder='drop')


pipe_preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', ElasticNet(alpha=0.01, l1_ratio=0.9))
])

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# Evaluate the pipeline using cross-validation
scores = cross_val_score(pipe_preprocessing_model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

pipe_preprocessing_model.fit(X,y)

rmse_scores = np.sqrt(-scores) 

mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

print('Mean RMSE: %.3f' % mean_rmse)
print('Standard Deviation of RMSE: %.3f' % std_rmse)

pickle.dump(pipe_preprocessing_model, open("price_model.pkl","wb"))






