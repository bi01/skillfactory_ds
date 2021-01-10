import ast
import datetime 
import numpy as np
import pandas as pd

from more_itertools import flatten
from pandas.api.types import CategoricalDtype


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



class CleanerDatasetPipeline():
    """ Cleaning, adding features and transformating dataset
    
        Using as separate steps, as all steps in pipeline
    """
    
    def __init__(self, df: pd.DataFrame) -> None:
        """
        """
        self.df = df
        
        self.process_pipeline = (
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
            
            Removing `$` character, changing to categorial type for this column
        """
        price_range_map = {
            np.NaN: 2,
            '$': 1,
            '$$ - $$$': 2,
            '$$$$': 3
        }

        self.df['price_range'] = self.df.price_range.replace(to_replace=price_range_map)
        self.df['price_range'] = self.df.price_range.astype(
            CategoricalDtype(categories=[1, 2, 3], ordered=True)
        )
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
        
        # create new column
        self.df['timedelta_reviews'] = self.df.review_date1 - self.df.review_date2
        
    
    def pipeline_desciption(self) -> None:
        """ Show doc info by each step/process in pipeline
        """
        for item_number, process in enumerate(self.process_pipeline):
            print(item_number, process.__doc__.split('\n', 1)[0])
        
    def run_pipeline(self) -> pd.DataFrame:
        """ Run pipeline during which transformations are applied for specified data frame
        """
        for process in self.process_pipeline:
            process()
        
        return self.df
        