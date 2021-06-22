# imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.data import get_data, clean_data, hold_out
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
import pandas as pd


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
            ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipe

    def run(self):
        """set and train the pipeline"""
        #Create the pipeline
        self.pipeline = self.set_pipeline()
        
        #Fit the pipeline
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data()
    
    # clean data
    cleaned_df = clean_data(df)
    
    # set X and y
    y = df.pop("fare_amount")
    X = df
    
    # hold out
    X_train, X_test, y_train, y_test = hold_out(X,y)
    
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    
    # evaluate
    score = trainer.evaluate(X_test, y_test)
    
    print(f"RMSE: {score}")
