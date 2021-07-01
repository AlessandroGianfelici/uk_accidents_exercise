import pandas as pd
import numpy as np
import logging
from pampy import match, _
from functools import reduce, partial
from operator import add
import holidays
import numpy as np
import pandas as pd
import plotly.express as px
logger = logging.getLogger("__main__")

def plotHistogram(feature, train_set):
    x0 = train_set.loc[train_set['GROUND_TRUTH'] == 0][feature]
    x1 = train_set.loc[train_set['GROUND_TRUTH'] == 1][feature]

    df =pd.DataFrame(dict(
        series=np.concatenate((["No police officer attended"]*len(x0), ["A police officer attended"]*len(x1))), 
        data  =np.concatenate((x0,x1))
    ))

    fig = px.histogram(df, x="data", color="series", 
                 barmode="overlay", 
                 histnorm='probability density',
                 title = f"Exploring feature {feature}"
                 )
    fig.update_layout(bargap=0.1)
    return fig


class FeatureExtraxtor(object):

    def __init__(self, url):
        self._data = pd.read_csv(url, parse_dates=['Date'])
        self._perimetro = self._data['Accident_Index'].drop_duplicates()
        return

    @property
    def data(self):
        return self._data

    def getGroundTruth(self):
        gt = self.data[['Accident_Index', 'Did_Police_Officer_Attend_Scene_of_Accident']]
        gt['GROUND_TRUTH'] = (gt['Did_Police_Officer_Attend_Scene_of_Accident'] == 1).astype(int)
        return gt[['Accident_Index', 'GROUND_TRUTH']]

    def getFeatures(self):
        """
        This function merges all the features toghether and fill the missing values.
        The choice to compute every feature group using a different function and to put 
        all togheter at the end has the advantage of making easy to add/remove new features
        from the etl, but is inefficient from a performance point of view.
        I've chosen this method because of the dimension of the dataset, with a larger dataset
        the best choice would be to perform all the operation in place on the original dataframe.
        """ 
        # N.B. I will pass self.data instead of self._data for safety reason (it is readonly)
        features = self.mergeFeatures([
            self._perimetro,
            #self.getHoliday(self.data),
            self.othersNumericalFeatures(self.data),
            self.getRoadDimension(self.data),
            self.getSurfaceConditions(self.data),
            self.getWeekend(self.data)]).set_index('Accident_Index')

        for col in features.columns:
            features[col] = features[col].fillna((features[col].mean()))

        # I will delete now some irrelevant features (they have importances=0 according to the feature importances graph)
        features = features.drop(columns=['is_from_Oceania','language_other','is_from_Africa',
                                          'product_free_trial_length','iPod'], errors='ignore')
        return features.reset_index()

    @staticmethod
    def othersNumericalFeatures(data):
        """
        This function returns the columns of the input dataframe that can be used by the model without any proprocessing.
        """
        logger.info("*** getting numerical features...")

        return data[['Accident_Index',
                    'Latitude',
                    'Longitude',
                    'Number_of_Vehicles',
                    'Number_of_Casualties',
                    'Accident_Severity',
                    'Road_Type',
                    'Speed_limit',
                    'Junction_Detail',
                    'Junction_Control',
                    'Pedestrian_Crossing-Physical_Facilities',
                    'Light_Conditions',
                    'Weather_Conditions',
                    'Special_Conditions_at_Site',
                    'Carriageway_Hazards',
                    'Urban_or_Rural_Area']]

    @staticmethod
    def getWeekend(data):
        data['isWeekend'] = (data['Day_of_Week'] == 1) | (data['Day_of_Week'] == 7).astype(int)
        return data[['Accident_Index', 'isWeekend']]

    @staticmethod
    def getRoadDimension(data):
        myDict = {1 : 1,
                  2 : 1,
                  3 : 1,
                  4 : 2,
                  5 : 2,
                  6 : 3}
        data['RoadDimension'] = data['1st_Road_Class'].map(myDict)
        return data[['Accident_Index', 'RoadDimension']]

    @staticmethod
    def getSurfaceConditions(data):
        myDict = {1 : 1,
                  -1 : 0.5,
                  2 : 0,
                  3 : 0,
                  4 : 0,
                  5 : 0,
                  6 : 0,
                  7 : 0}
        data['SurfaceConditions'] = data['Road_Surface_Conditions'].map(myDict)
        return data[['Accident_Index', 'SurfaceConditions']]

    def getHoliday(self, data):
        """
        This function returns a binary features which is 1 if the user started the free trial on holiday, 0 elsewhere
        """
        logger.info("*** getting calendar day features...")
        data['is_holiday'] = data.apply(self.isHolidays, axis=1)
        return data[['Accident_Index', 'is_holiday']]

    @staticmethod
    def isHolidays(df):
        """
        This function returns 1 if the free trial has been activated on holiday, 0 elsewhere.
        """
        try:
            # I am using the holidays package. Not every country is supported, so if a country is missing, I simply consider justweekend as holiday.
            holidaysInTheYear = holidays.CountryHoliday('uk', years = df['Date'].year)
            is_holiday = (df['Date'] in holidaysInTheYear)
        except:
            is_holiday = False
        return int(is_holiday)

    def mergeFeatures(self, data_frames):
        """
        This method take as input a list of dataframes and return their outer join on the column 'Accident_Index'.
        """
        return reduce(lambda left, right: pd.merge(left,right,on='Accident_Index', how='outer'), data_frames)