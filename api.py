import numpy as np
import pickle
import pandas as pd
import datetime
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import performance_metrics

prophet_final = pickle.load(open('Models/Prophet_Final.pkl', 'rb'))
df = pd.read_pickle("Data/Prophet_Forecast_DF.pkl")

example = {
    'Steps': df.describe()['Steps']['mean'],
    'Minutes_Sedentary': df.describe()['Minutes_Sedentary']['mean'],
    'Minutes_Lightly_Active': df.describe()['Minutes_Lightly_Active']['mean'],
    'Minutes_Fairly_Active': df.describe()['Minutes_Fairly_Active']['mean'],
    'Minutes_Very_Active': df.describe()['Minutes_Very_Active']['mean'],
    'Calories_Breakfast': df.describe()['Calories_Breakfast']['mean'],
    'Saturated_Fat_Breakfast': df.describe()['Saturated_Fat_Breakfast']['mean'],
    'Cholesterol_Breakfast': df.describe()['Cholesterol_Breakfast']['mean'],
    'Polyunsaturated_Fat_Breakfast': df.describe()['Polyunsaturated_Fat_Breakfast']['mean'],
    'Monounsaturated_Fat_Breakfast': df.describe()['Monounsaturated_Fat_Breakfast']['mean'],
    'Trans_Fat_Breakfast': df.describe()['Trans_Fat_Breakfast']['mean'],
    'Carbohydrates_(g)_Breakfast': df.describe()['Carbohydrates_(g)_Breakfast']['mean'],
    'Sugar_Breakfast': df.describe()['Sugar_Breakfast']['mean'],
    'Sodium_(mg)_Breakfast': df.describe()['Sodium_(mg)_Breakfast']['mean'],
    'Protein_(g)_Breakfast': df.describe()['Protein_(g)_Breakfast']['mean'],
    'Calories_Lunch': df.describe()['Calories_Lunch']['mean'],
    'Saturated_Fat_Lunch': df.describe()['Saturated_Fat_Lunch']['mean'],
    'Polyunsaturated_Fat_Lunch': df.describe()['Polyunsaturated_Fat_Lunch']['mean'],
    'Monounsaturated_Fat_Lunch': df.describe()['Monounsaturated_Fat_Lunch']['mean'],
    'Trans_Fat_Lunch': df.describe()['Trans_Fat_Lunch']['mean'],
    'Cholesterol_Lunch': df.describe()['Cholesterol_Lunch']['mean'],
    'Sodium_(mg)_Lunch': df.describe()['Sodium_(mg)_Lunch']['mean'],
    'Carbohydrates_(g)_Lunch': df.describe()['Carbohydrates_(g)_Lunch']['mean'],
    'Sugar_Lunch': df.describe()['Sugar_Lunch']['mean'],
    'Protein_(g)_Lunch': df.describe()['Protein_(g)_Lunch']['mean'],
    'Calories_Dinner': df.describe()['Calories_Dinner']['mean'],
    'Saturated_Fat_Dinner': df.describe()['Saturated_Fat_Dinner']['mean'],
    'Polyunsaturated_Fat_Dinner': df.describe()['Polyunsaturated_Fat_Dinner']['mean'],
    'Monounsaturated_Fat_Dinner': df.describe()['Monounsaturated_Fat_Dinner']['mean'],
    'Trans_Fat_Dinner': df.describe()['Trans_Fat_Dinner']['mean'],
    'Cholesterol_Dinner': df.describe()['Cholesterol_Dinner']['mean'],
    'Sodium_(mg)_Dinner': df.describe()['Sodium_(mg)_Dinner']['mean'],
    'Carbohydrates_(g)_Dinner': df.describe()['Carbohydrates_(g)_Dinner']['mean'],
    'Sugar_Dinner': df.describe()['Sugar_Dinner']['mean'],
    'Protein_(g)_Dinner': df.describe()['Protein_(g)_Dinner']['mean'],
}


def prophet_forecast_weight(features, df, periods, date, cap=None, floor=170):
    future = prophet_final.make_future_dataframe(periods=periods)
    future['floor'] = 170

    steps_regressors_df = df[['Date', 'Steps', 'Minutes_Sedentary', 'Minutes_Lightly_Active',
                              'Minutes_Fairly_Active', 'Minutes_Very_Active', 'Calories_Breakfast',
                              'Saturated_Fat_Breakfast', 'Cholesterol_Breakfast',
                              'Polyunsaturated_Fat_Breakfast', 'Monounsaturated_Fat_Breakfast',
                              'Trans_Fat_Breakfast', 'Carbohydrates_(g)_Breakfast', 'Sugar_Breakfast',
                              'Sodium_(mg)_Breakfast',
                              'Protein_(g)_Breakfast', 'Calories_Lunch', 'Saturated_Fat_Lunch',
                              'Polyunsaturated_Fat_Lunch', 'Monounsaturated_Fat_Lunch',
                              'Trans_Fat_Lunch', 'Cholesterol_Lunch', 'Sodium_(mg)_Lunch',
                              'Carbohydrates_(g)_Lunch', 'Sugar_Lunch', 'Protein_(g)_Lunch',
                              'Calories_Dinner', 'Saturated_Fat_Dinner', 'Polyunsaturated_Fat_Dinner',
                              'Monounsaturated_Fat_Dinner', 'Trans_Fat_Dinner', 'Cholesterol_Dinner',
                              'Sodium_(mg)_Dinner', 'Carbohydrates_(g)_Dinner',
                              'Sugar_Dinner', 'Protein_(g)_Dinner']]

    columns = ['Steps', 'Minutes_Sedentary', 'Minutes_Lightly_Active',
               'Minutes_Fairly_Active', 'Minutes_Very_Active', 'Calories_Breakfast',
               'Saturated_Fat_Breakfast', 'Cholesterol_Breakfast',
               'Polyunsaturated_Fat_Breakfast', 'Monounsaturated_Fat_Breakfast',
               'Trans_Fat_Breakfast', 'Carbohydrates_(g)_Breakfast', 'Sugar_Breakfast',
               'Sodium_(mg)_Breakfast',
               'Protein_(g)_Breakfast', 'Calories_Lunch', 'Saturated_Fat_Lunch',
               'Polyunsaturated_Fat_Lunch', 'Monounsaturated_Fat_Lunch',
               'Trans_Fat_Lunch', 'Cholesterol_Lunch', 'Sodium_(mg)_Lunch',
               'Carbohydrates_(g)_Lunch', 'Sugar_Lunch', 'Protein_(g)_Lunch',
               'Calories_Dinner', 'Saturated_Fat_Dinner', 'Polyunsaturated_Fat_Dinner',
               'Monounsaturated_Fat_Dinner', 'Trans_Fat_Dinner', 'Cholesterol_Dinner',
               'Sodium_(mg)_Dinner', 'Carbohydrates_(g)_Dinner',
               'Sugar_Dinner', 'Protein_(g)_Dinner']

    future = future.merge(steps_regressors_df, left_on='ds', right_on='Date', how='left')

    for i in columns:
        future[i].iloc[-periods:] = features[i]

    forecast = prophet_final.predict(future)

    predicted_weight = np.round(forecast[forecast['ds'] == date]['yhat'].values[0], 2)
    predicted_weight_upper = np.round(forecast[forecast['ds'] == date]['yhat_upper'].values[0], 2)
    predicted_weight_lower = np.round(forecast[forecast['ds'] == date]['yhat_lower'].values[0], 2)

    result = {

        'predicted_weight': predicted_weight,
        'predicted_weight_upper': predicted_weight_upper,
        'predicted_weight_lower': predicted_weight_lower

    }

    return result


if __name__ == '__main__':
    print(prophet_forecast_weight(example))
