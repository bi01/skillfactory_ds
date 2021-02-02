import ast
import datetime 

import numpy as np
import pandas as pd

from more_itertools import flatten
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics 


def convert_cuisine(row: [str, np.nan]) -> [list, np.nan]:
    """ Convert string representation of list to list
        np.nan values leaves as it is
    """
    if not isinstance(row, list) and pd.notna(row):
        return [x.strip() for x in ast.literal_eval(row)]

    return row


def convert_reviews(reviews_data: [str, np.nan, list]) -> list:
    """ Convert the reviews record to 4 records 
        
        Returned format: 
            [date_review1, review1, date_review2, review2]
    """
    
    if not isinstance(reviews_data, list) and pd.notna(reviews_data):
        
        # convert list string literal to list structure
        reviews_list = [x for x in ast.literal_eval(reviews_data.replace('nan', 'None'))]

        # checking what it's empty nested list 
        if not list(flatten(reviews_list)):
            return [np.nan, np.nan, np.nan, np.nan]
        
        # making the pair or a pairs in list - review + date 
        result = list(flatten(zip(reviews_list[0], reviews_list[1])))
        
        # returning always 4 and, even if you received 2 or 4 records
        return result + [np.nan, np.nan] if len(result) == 2 else result    

    # else return np.nan or already processing value
    return reviews_data


def convert_datetime(cell):
    """ Convert to datetime format a string datetime literal 
    """
    if pd.notna(cell):
        return datetime.datetime.strptime(cell, '%m/%d/%Y')
    
    return cell





class ModelProcessing():
    """ Dividing and preparing model for learning
    """
    def __init__(self, df) -> None:
        self.df = df
        
        # the order of steps is important!
        self.pipeline = (
            self.get_Xy,
            self.get_train_test,
            self.training_model,
            self.get_MAE
        )
              
    def get_Xy(self) -> dict[pd.DataFrame, pd.Series]:
        """ Getting 2 pieces - data as learning dataset and vector of a target variable 
            
            where:
                self.Х - data with info by restaurant, 
                self.у - target variable (ranking of restaurants)
        """
        self.X = self.df.drop(['restaurant_id', 'rating'], axis = 1)  
        self.y = self.df['rating'] 
        
#         return {
#             'X': self.X,
#             'y': self.y
#         }

    def get_train_test(self) -> dict[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """ Separating dataframe on pieces, needed for the learning and testing of the model

            Data sets with label:
            - "train" - will be using for a learning of the module, 
            - "test" - for only testing.  
            - before separating we remove useless columns from train/test dataset

            For the testing we will be using 25% from source data set.  
        """
        (self.X_train, 
         self.X_test, 
         self.y_train, 
         self.y_test) = train_test_split(self.X, self.y, test_size=0.25)       
        
        dropped_columns =  [
            'city', 
            'cuisine_style', 
            'url_ta', 
            'id_ta',
            'review1',
            'review_date1',
            'review2', 
            'review_date2',
            'timedelta_reviews'
        ]
        
        self.X_train = self.X_train.drop(dropped_columns, axis=1).fillna(0)
        self.X_test = self.X_test.drop(dropped_columns, axis=1).fillna(0)
        
#         return {
#             'X_train': self.X_train, 
#             'X_test': self.X_test, 
#             'y_train': self.y_train,
#             'y_test': self.y_test
#         }       

    def training_model(self) -> np.array:
        """ Create, learning, predicting result for a model
        """
        self.regr = RandomForestRegressor(n_estimators=100)  
        self.regr.fit(self.X_train, self.y_train)  
        self.y_pred = self.regr.predict(self.X_test)

        return self.y_pred

    def get_MAE(self) -> float:
        """ Returning the MAE indicator
        """
        return metrics.mean_absolute_error(self.y_test, self.y_pred)
        
    def show_steps(self) -> None:
        """ Show doc info by each step/process in pipeline
        """
        for item_number, process in enumerate(self.pipeline):
            print(item_number, '-', process.__doc__.split('\n', 1)[0])
        
    def run(self) -> None:
        """ Running all steps for preraring of a model
        """  
        for item_number, process in enumerate(self.pipeline):
            result = process()
            doc_info =  process.__doc__.split('\n', 1)[0]
            print(item_number, doc_info, ':', result)
     
            
class DatasetCleaner():
    """ Cleaning, adding features and transformating dataset
    
        Using as separate steps, as all steps in pipeline
    """
    
    def __init__(self, df: pd.DataFrame) -> None:
        """ Setting dataframe and pipeline when creating example of this class
        """
        self.df = df
        
        # the order of steps is important!
        self.pipeline = (
            self.change_columns,
            self.processing_price_range,
            self.processing_city,
            self.processing_cuisine_style,
            self.processing_reviews,
        )
  
    def change_columns(self) -> pd.DataFrame:
        """ We give more understandable names of the data frame columns
        """
        self.df.columns = [
            'restaurant_id',
            'city',
            'cuisine_style',
            'ranking',
            'rating',
            'price_range',
            'number_of_reviews',
            'reviews',
            'url_ta',
            'id_ta'
        ]
        return self.df
    
    def processing_price_range(self) -> pd.DataFrame:
        """ Processing `price range` column

            - replace Nan values - they split equally into the remaining three categories
            - Removing `$` character, 
            - changing to categorial type for this column

        """
        # раскидаем поровну на три категории все пропущенные значения, 
        # сохраняя пропорции между ними 
        na_part_values = len(self.df[self.df.price_range.isnull()]) // 3

        for x in self.df[self.df.price_range.isnull()][:na_part_values].index:
            self.df.at[x, 'price_range'] = '$'

        for x in self.df[self.df.price_range.isnull()][:na_part_values].index:
            self.df.at[x, 'price_range'] = '$$$$'

        self.df.price_range.fillna(value='$$ - $$$', inplace=True)
        len(self.df[self.df.price_range.isna()])

        # заменим значения на целочисленные категории 
        price_range_map = {
            '$': 'cheap',
            '$$ - $$$': 'average',
            '$$$$': 'expensive'
        }
        self.df['price_range'] = self.df.price_range.map(price_range_map)

        # устанавливаем категории в соотвествии: cheap-$, average-$$-$$$, expensive-$$$$
        pd.Categorical(self.df.price_range, ['cheap', 'average', 'expensive'], ordered=True)
        
        # dummy переменные для значений категорий price_range
        data_dummies = pd.get_dummies(self.df.price_range, prefix='price')
        self.df = self.df.merge(data_dummies, left_index=True, right_index=True)
        self.df.drop(['price_range'], axis=1, inplace=True)

        return self.df
    
    def processing_city(self) -> pd.DataFrame:
        """ Processing `city` column 
        """
        return self.df
    
    def processing_cuisine_style(self) -> pd.DataFrame:
        """ Processing `cuisine_style` column and added new column `total_types_cuisine`
        """
        self.df['cuisine_style'] = self.df.cuisine_style.apply(convert_cuisine)
        self.df['total_types_cuisine'] = self.df.cuisine_style.fillna(1).apply(lambda x: 1 if isinstance(x, int) else len(x))
        
        return self.df

    def processing_reviews(self) -> pd.DataFrame:
        """ Processing `reviews` column, 1 column remove, 5 new columns adding
        
            - Create 4 columns: 'review1', 'review_date1', 'review2', 'review_date2'
            - Remove `reviews` columns 
            - Create `timedelta_reviews` columns
        """
        reviews_df = pd.DataFrame(
            data=list(self.df.reviews.apply(convert_reviews)), 
            columns=['review1', 'review_date1', 'review2', 'review_date2']
        )
        reviews_df['review_date1'] = reviews_df.review_date1.apply(convert_datetime)
        reviews_df['review_date2'] = reviews_df.review_date2.apply(convert_datetime)

        # create new 4 columns concating with the original data frame
        self.df = pd.concat([self.df, reviews_df], axis=1)
        
        # dropping useless now a `reviews` column
        self.df.drop(['reviews'], axis=1, inplace=True) 
        
        # create new column `timedelta_reviews` and filling nan values
        self.df['timedelta_reviews'] = self.df.review_date1 - self.df.review_date2
        self.df.timedelta_reviews.fillna(pd.Timedelta(seconds=0))
        
    def show_steps(self) -> None:
        """ Show doc info by each step/process in pipeline
        """
        for item_number, process in enumerate(self.pipeline):
            print(item_number, '-', process.__doc__.split('\n', 1)[0])
        
    def run(self) -> pd.DataFrame:
        """ Run pipeline during which transformations are applied for specified data frame
        """
        for process in self.pipeline:
            process()
        
        return self.df.copy()
        