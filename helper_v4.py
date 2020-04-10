
import pandas as pd
import numpy as np
from flask_socketio import SocketIO, emit
import time
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import mean_absolute_error,mean_squared_error
from statsmodels.tsa import arima_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from copy import deepcopy
import joblib
from sklearn.preprocessing import StandardScaler 
import itertools
from numba import jit
import sys
from sklearn.externals import joblib
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import datetime
import os
import argparse
from itertools import product
import glob
np.random.seed(0)

import logging
logging.captureWarnings(True)
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

def forecastr(data,forecast_settings,column_headers,freq_val,build_settings):

    """
    Background: This function will take the data from the csv and forecast out x number of days.

    Input:

    data: This is a pandas dataframe containing time series data, datetime first column
    forecast_settings: This is a list containing values for model type, forecast period length,test_period and seasonality parameters
    column_headers: List containing the name of the date and metric
    freq_val: String containing "D","M","Y"
    build_settings: String determining whether this is an initial or updated forecast.


    Output:

    [y_hat,dates,m,csv_ready_for_export]: A list containing forecasted data, dimension, model and data for the csv export


    """


    ##### Variables, Model Settings & Facebook Prophet Hyper Parameters #####

    # Initial Variables
    build = build_settings                                  # Determine the build_setting - either initial or update forecast settings.
    dimension = column_headers[0]                           # date
    metric = column_headers[1]                              # metric name

    # Rename the columns so we can use FB Prophet
    data.rename(columns={dimension: "date", metric: "y"}, inplace=True)
   
    
    # Hyper-parameters
    fs_model_type = forecast_settings[0]                    # linear or logistic
    fs_forecast_period = int(forecast_settings[1])                   # forecast period
    fs_test_period=int(forecast_settings[2])# test period
    if fs_model_type=="Moving_Average":
        my_type="ma"
    elif fs_model_type=="SARIMA":
        my_type="sarima"

   
    d = range(0,2)
    p  = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    m_1= range(0,13)
    seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q,m_1))]
    pdq = pdq[1:]
    
    # Instantiate with prophet_arg_vals that are not auto, 0 or False.
    model=prediction_func(data,pdq=pdq,seasonal_pdq=seasonal_pdq,test_day=fs_test_period,model_type=my_type)


    # Status update
    emit('processing', {'data': 'model has been fit'})


    # Let's create a new data frame for the forecast which includes how long the user requested to forecast out in time units and by time unit type (eg. "D", "M","Y")
    #future = m.make_future_dataframe(periods=fs_period, freq=freq_val)

    # If fs_model_type = 'logistic', create a column in future for carrying_capacity and saturated_minimum
    '''
    if fs_model_type == 'logistic':
        future['cap'] = fs_carrying_capacity
        future['floor'] = fs_saturated_minimum
    else:
        print('no cap or floor needed as it is a linear model.')
'''
    # Let's predict the future :)
    y_forecast = model.forecast(fs_forecast_period+1).tolist()
    y_hat=model.predict().tolist()
    preds=y_hat+y_forecast
    ##### Send y_hat and dates to a list, so that they can be graphed easily when set in ChartJS
    data_new=data.append(pd.DataFrame({"date": pd.date_range(start=data.date.iloc[-1], periods=fs_forecast_period)}))
    data_new["prediction"]=preds
    data_new["yhat_upper"]=preds
    data_new["yhat_lower"]=preds
    #y_hat = data_new['preds'].tolist()
    dates = data_new['date'].apply(lambda x: str(x).split(' ')[0]).tolist()

    ##### Lets see how the forecast compares to historical performance #####

    # First, lets sum up the forecasted metric
    forecast_sum = sum(y_hat)
    forecast_mean = np.mean(y_hat)



    # Now lets sum up the actuals for the same time interval as we predicted
    actual_sum = data_new["y"].sum()
    actual_mean = data_new["y"].mean()

    difference = '{0:.1%}'.format(((forecast_sum - actual_sum) / forecast_sum))
    difference_mean = '{0:.1%}'.format(((forecast_mean - actual_mean) / forecast_mean))


    forecasted_vals = ['{0:.1f}'.format(forecast_sum),'{0:.1f}'.format(actual_sum),difference]
    forecasted_vals_mean = ['{0:.1f}'.format(forecast_mean),'{0:.1f}'.format(actual_mean),difference_mean]

    



  

    ####### Formatting data for CSV Export Functionality ##########


    # First, let's merge the original and forecast dataframes
    #data_for_csv_export = pd.merge(forecast,data,on='date',how='left')

    # Select the columns we want to include in the export
    #export_formatted = data_for_csv_export[['ds','y','yhat','yhat_upper','yhat_lower']]
    
    # Rename y and yhat to the actual metric names
    data_new.rename(index=str, columns={'date': 'date', 'y': metric, 'yhat': metric + '_forecast','yhat_upper':metric + '_upper_forecast','yhat_lower':metric + '_lower_forecast'}, inplace=True)

    # replace NaN with an empty val
    data_new = data_new.replace(np.nan, '', regex=True)

    # Format timestamp
    data_new['date'] = data_new['date'].apply(lambda x: str(x).split(' ')[0])

    # Create dictionary format for sending to csv
    #csv_ready_for_export = export_formatted.to_dict('records')
    csv_ready_for_export = data_new.to_dict('records')
    

    # print(y_hat)
    # print(csv_ready_for_export)
    print(forecasted_vals)
    print(forecasted_vals_mean)
    print(preds)
    print(model)
    return [preds,dates,model,csv_ready_for_export,forecasted_vals, forecasted_vals_mean,data_new]



def validate_model(model,dates):

    """

    Background:

    This model validation function is still under construction and will be updated during a future release.


    """

    count_of_time_units = len(dates)
    #print(count_of_time_units)
    initial_size = str(int(count_of_time_units * 0.20)) + " days"
    horizon_size = str(int(count_of_time_units * 0.10)) + " days"
    period_size = str(int(count_of_time_units * 0.05)) + " days"

    df_cv = cross_validation(model, initial=initial_size, horizon=horizon_size, period=period_size)
    #df_cv = cross_validation(model,initial='730 days', period='180 days', horizon = '365 days')
    df_p = performance_metrics(df_cv)

    #print(df_cv.head(100))
    #print(df_p.head(100))

    mape_score_avg = str(round(df_p['mape'].mean()*100,2)) + "%"

    return mape_score_avg




def check_val_of_forecast_settings(param):

    """

    Background:

    This function is used to check to see if there is a value (submitted from the user in the UI) for a given Prophet Hyper Parameter. If there is no value or false or auto, return that, else we'll return a float of the param given that the value may be a string.

    If the param value is blank, false or auto, it will eventually be excluding from the dictionary being passed in when instantiating Prophet.

    """


    # Check hyper parameter value and return appropriate value.
    if (param == "") or (param == False) or (param == 'auto'):
        new_arg = param
        return new_arg

    else:
        new_arg = float(param)
        return new_arg





def get_summary_stats(data,column_headers):

    """

    Background:
    This function will get some summary statistics about the original dataset being uploaded.

    Input:

    data: a dataframe with the data from the uploaded csv containing a dimension and metric
    column_headers: string of column names for the dimension and metric


    Output:

    sum_stats: a list containing the count of time units, the mean, std, min and max values of the metric. This data is rendered on step 2 of the UI.

    """

    # Set the dimension and metrics
    dimension = column_headers[0]
    metric = column_headers[1]



    time_unit_count = str(data[dimension].count())





    print(data[metric].mean())

    mean = str(round(data[metric].mean(),2))
    print('string of the mean is ' + mean)


    std = str(round(data[metric].std(),2))
    minimum = str(round(data[metric].min(),2))
    maximum = str(round(data[metric].max(),2))

    sum_stats = [time_unit_count,mean,std,minimum,maximum]
    print(sum_stats)

    return sum_stats




def preprocessing(data):


    """

    Background: This function will determine which columns are dimensions (time_unit) vs metrics, in addition to reviewing the metric data to see if there are any objects in that column.

    Input:

        data (df): A dataframe of the parsed data that was uploaded.

    Output:

        [time_unit,metric_unit]: the appropriate column header names for the dataset.

    """

    # Get list of column headers
    column_headers = list(data)


    # Let's determine the column with a date

    col1 = column_headers[0]
    col2 = column_headers[-1] #last column
    print('the first column is ' + col1)
    print("target column is" +col2)
    # Get the first value in column 1, which is what is going to be checked.
    col1_val = data[col1][0]
    print(type(col1_val))
    print(data.shape)
  

    # Check to see if the data has any null values

    #print('Is there any null values in this data? ' + str(data.isnull().values.any()))

    # If there is a null value in the dataset, locate it and emit the location of the null value back to the client, else continue:

    #print(data.tail())
    print('Is there any null values in this data? ' + str(data.isnull().values.any()))
    do_nulls_exist = data.isnull().values.any()

    if do_nulls_exist == True:
        print('found a null value')
        null_rows = pd.isnull(data).any(1).nonzero()[0]
        #print('######### ORIGINAL ROWS THAT NEED UPDATING ##############')
        #print(null_rows)
        # Need to add 2 to each value in null_rows because there

        #print('######### ROWS + 2 = ACTUAL ROW NUMBERS IN CSV ##############')
        update_these_rows = []
        for x in null_rows:
            update_these_rows.append(int(x+2))
        
        print(update_these_rows)
        emit('error', {'data': update_these_rows})
        data=data.fillna(method="bfill")




    else:
        print('no nulls found')


    if isinstance(col1_val, (int, np.integer)) or isinstance(col1_val, float):
        print(str(col1_val) + ' this is a metric')
        print('Setting time_unit as the second column')
        time_unit = column_headers[-1]
        metric_unit = column_headers[0]
        return [time_unit, metric_unit]
    else:
        print('Setting time_unit as the first column')
        time_unit = column_headers[0]
        metric_unit = column_headers[-1]
        return [time_unit, metric_unit]






def determine_timeframe(data, time_unit):

    """

    Background:

    This function determines whether the data is daily, weekly, monthly or yearly by checking the delta between the first and second date in the df.

    Input:

    data: a df containg a dimension and a metric
    time_unit: is the dimension name for the date.


    Output:

    time_list: a list of strings to be used within the UI (time, desc) and when using the function future = m.make_future_dataframe(periods=fs_period, freq=freq_val)



    """


    # Determine whether the data is daily, weekly, monthly or yearly
    date1 = data[time_unit][0]
    date2 = data[time_unit][1]

    first_date = pd.Timestamp(data[time_unit][0])
    second_date = pd.Timestamp(data[time_unit][1])
    time_delta = second_date - first_date

    time_delta = int(str(time_delta).split(' ')[0])

    print([data[time_unit][0],data[time_unit][1]])
    print([second_date,first_date,time_delta])


    if time_delta == 1:
        time = 'days'
        freq = 'D'
        desc = 'daily'
    elif time_delta >=7 and time_delta <= 27:
        time = 'weeks'
        freq = 'W'
        desc = 'weekly'
    elif time_delta >=28 and time_delta <=31:
        time = 'months'
        freq = 'M'
        desc = 'monthly'
    elif time_delta >= 364:
        time = 'years'
        freq = 'Y'
        desc = 'yearly'
    else:
        print('error?')

    time_list = [time,freq, desc]
    #print(time_list)

    return time_list



def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates the mean absolute percentage error.
    Args:
    y_true: (np.array) actual values
    y_pred: (np.array) predicted values
    
    Returns: 
    float value"""
    

    return np.mean(np.abs((y_true +0.00001*np.random.rand(len(y_true)) - y_pred+0.00001*np.random.rand(len(y_true))) / y_true+0.00001*np.random.rand(len(y_true)))) * 100


def predict_sarimax_model(x_train,y_train,x_test,y_test,order,seasonal_order,feats_to_use=None,round_predictions=False,plot_results=True):
    
    #ipdb.set_trace()
    """
    Predicts univariate predictions of data.
    
    Args:
    x_train: (pandas dataframe) 
    y_train: (pandas dataframe)
    x_test: (pandas dataframe) 
    y_test: (pandas dataframe)
    order: (tuple) (p,d,q)
    seasonal_order:(tuple) (P,D,Q,S)
    feats_to_use: (list) exog features to use.
    
    Returns: 
    model_fit: (statsmodels object)
    predictions: (pandas dataframe)
    """
    
    predictions = pd.Series()
    y_train_history = y_train.copy()
    
    if feats_to_use is None:
        feats_to_use = x_train.columns
        
    x_train_history = x_train[feats_to_use].copy()
    for t in pd.DataFrame(y_test).iterrows():
        model = SARIMAX(endog=y_train_history,exog=x_train_history[feats_to_use],order=order,seasonal_order=seasonal_order,
                        enforce_stationarity=False,enforce_invertibility=False)
        
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(exog=pd.DataFrame(x_test[feats_to_use].loc[t[0],:]).T)
       
        if output.iloc[0]< y_train.min() :
            yhat = y_train.min()
        elif output.iloc[0]> y_train.max() :
            yhat =y_train.max()
        else :
            if round_predictions:
                yhat = round(output.iloc[0],0)
            else:
                yhat = output.iloc[0]
                
        
        if round_predictions:
            predictions.loc[t[0]] = round(yhat,0)
        else:
            predictions.loc[t[0]] = yhat
            
        y_train_history.loc[t[0]] = t[1].values[0]
        x_train_history.loc[t[0],:] = x_test[feats_to_use].loc[t[0],:]
        #x_train_history  = pd.concat([x_train_history[feats_to_use],
        #                              pd.DataFrame(x_test[feats_to_use].loc[t[0],:]).T],axis=0)
        #print("Period: ",t[0],'predicted=%f, expected=%f' % (yhat,  t[1].values[0]))
        #print("Period: ",t[0],'predicted=%f, expected=%f' % (yhat,  t[1].values[0]))
    
    try:
        
        if plot_results:
            mse_error = np.sqrt(mean_squared_error(y_test, predictions))
            mae_error = mean_absolute_error(y_test, predictions)
            mape_error = mean_absolute_percentage_error(y_test.values, predictions.values)
            
            print('Test RMSE: %.3f' % mse_error)
            print('Test MAE: %.3f' % mae_error)
            print('Test MAPE: %.3f' % mape_error)
        # plot
        predictions = pd.DataFrame(data=y_test.values,index=y_test.index,columns=['actual']).merge(pd.DataFrame(data=predictions.values,
                                                                                            index=predictions.index,columns=['predictions']),
                                                                                            left_index=True,right_index=True)
        predictions.plot()
        plt.show()
        print(model_fit.summary())
        
        
    except Exception as e:
        print(e)
        
    return model_fit,predictions


                        
def predict_sarima_model(train_set,test_set,order,seasonal_order,round_predictions=False,trend=None,plot_results=True):
    """
    Predicts univariate predictions of data.
    
    Args:
    train_set: (pandas dataframe) 
    test_set: (pandas dataframe)
    order: (tuple) (p,d,q)
    seasonal_order:(tuple) (P,D,Q,S)
    
    
    Returns: 
    model_fit: (statsmodels object)
    predictions: (pandas dataframe)
    """
    
    predictions = pd.Series()
    historical_data = train_set.copy()

    for t in pd.DataFrame(test_set).iterrows():
        model = SARIMAX(historical_data,order=order,seasonal_order=seasonal_order, \
                        enforce_stationarity=False,enforce_invertibility=False,trend=trend)
        
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()

        if output.iloc[0]< historical_data.min() :
            yhat = historical_data.min()

        elif output.iloc[0]> historical_data.max() :
            yhat =historical_data.max()

        else :
            if round_predictions:
                yhat = round(output.iloc[0],0)
            else:
                yhat = output.iloc[0]
                
        if round_predictions:
            predictions.loc[t[0]] = round(yhat,0)
        else:
            predictions.loc[t[0]] = yhat
        historical_data.loc[t[0]] = t[1].values[0]

    predictions = pd.DataFrame(predictions,columns=['predictions']).merge(pd.DataFrame(test_set.values,index=test_set.index,columns=['actual']),left_index=True,right_index=True,how='left')

    if plot_results:
        mse_error = mean_squared_error(predictions['actual'], predictions['predictions'])
        mae_error = mean_absolute_error(predictions['actual'], predictions['predictions'])
        mape_error = mean_absolute_percentage_error(predictions['actual'].values, predictions['predictions'].values)

        print('Test MSE: %.3f' % mse_error)
        print('Test MAE: %.3f' % mae_error)
        print('Test MAPE: %.3f' % mape_error)
        print(model_fit.summary())
        
    predictions = pd.DataFrame(predictions,columns=['predictions']).merge(pd.DataFrame(test_set.values,index=test_set.index,columns=['actual']),left_index=True,right_index=True,how='left')
    return model_fit,predictions

@jit
def grid_search_func(training_set,test_set,pdq,seasonal_pdq):
    grid_results = pd.DataFrame(columns=['aic','param_significance','mape','mae'])

    for tmp_pdq in pdq:
        for tmp_s_pdq in seasonal_pdq:
            tmp_model,tmp_pred_sarima = predict_sarima_model(train_set=training_set,
                                                    test_set=test_set,
                                                    order=tmp_pdq,seasonal_order= tmp_s_pdq ,plot_results = False)
            
            mae_error = mean_absolute_error(tmp_pred_sarima.actual, tmp_pred_sarima.predictions)
            mape_error = mean_absolute_percentage_error(tmp_pred_sarima.actual.values, tmp_pred_sarima.predictions.values)
            
            res_key =  str(tmp_pdq) +"|" +str(tmp_s_pdq)
            grid_results.loc[res_key,'aic'] = tmp_model.aic
            grid_results.loc[res_key,'param_significance'] = (tmp_model.pvalues <0.1).all()
            grid_results.loc[res_key,'mape'] = mape_error
            grid_results.loc[res_key,'mae'] = mae_error
            
    return grid_results


def multiprocessing(func, args,workers):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func, args)
    return list(res)

def prediction_step(train_data,validation_data,test_data,pdq,seasonal_pdq,model_type):
    #prediction_step(train_data,validation_data,test_data,pdq,seasonal_pdq,model_type='sarima')
    #***********************************************************************************
    print("pred step başı")
    print("model_type===",model_type)
    print("type of model_type object:",type(model_type))
    if model_type=="sarima":
        print("güzel gidiyor")
     ## Find optimal params
    try:
        grid_search_results = grid_search_func(training_set=train_data,
                               test_set=test_data,pdq=pdq,seasonal_pdq=seasonal_pdq).sort_values(by=['aic','mape'])

        grid_pdq = ast.literal_eval(grid_search_results.index[3].split("|")[0])
        seasonal_grid_pdq = ast.literal_eval(grid_search_results.index[3].split("|")[1])
    except:
        seasonal_grid_pdq = (0,0,0,0)
        grid_pdq = (1,0,0)
    
    
    
    #***********************************************************************************
    # Compare best prediction on validation set
    
    train_model = pd.concat([train_data,validation_data],axis=0)
    print(train_data.columns)
    if  True :#((train_data.shape[1]==1) & ((model_type=="sarima") | (model_type==None))): #use sarima
        print("AAAA")
        
        target_col=train_data.columns[0]
        print(target_col)
        
        valid_sarima_model,valid_pred = predict_sarima_model(train_set=train_model.loc[:,target_col],
                           test_set=validation_data.loc[:,target_col],
                                        order=grid_pdq,
                                        seasonal_order=seasonal_grid_pdq,
         
                                                                    plot_results = True,round_predictions=True)
           # check accuracy of predictions
        valid_pred_sarima=validation_data.copy()
        valid_pred_sarima.loc[:,"predictions"]=valid_pred
        mae_sarima = mean_absolute_error(valid_pred_sarima[target_col],valid_pred_sarima['predictions'])
        
        valid_ma = validation_data.copy()
        valid_ma.loc[:,'predictions'] = round(train_data.loc[:,target_col].tail(4).mean())
        mae_ma = mean_absolute_error(valid_ma[target_col],valid_ma['predictions'])
        d = {'sarima':mae_sarima,'ma':mae_ma}
        if model_type==None:
            model_type = min(d,key=d.get)
        print("I am in pred step -valid")
        print("model type is")
        print(model_type)
        print(valid_sarima_model)
    
    elif((train_data.shape[1]==1) & (model_type=="ma")): 
        target_col=train_data.columns[0]
        valid_ma = validation_data.copy()
        valid_ma.loc[:,'predictions'] = round(train_data.loc[:,target_col].tail(4).mean())
        mae_ma = mean_absolute_error(valid_ma[target_col],valid_ma['predictions'])
    else: #use SARIMAX 
        
        print("enter there")
        target_col=train_data.iloc[:,-1].name
        feature_cols= train_data.iloc[:,:-1]
        
        valid_sarimax_model,valid_pred_sarimax = predict_sarimax_model(x_train=train_model.loc[:,feature_cols],
                                                                   y_train=train_model.loc[:,target_col],
                                                                   x_test=validation_data.loc[:,feature_cols],
                                                                   y_test=validation_data.loc[:,target_col],
                                                                   order=grid_pdq, seasonal_order=seasonal_grid_pdq,
                                                                   feats_to_use=feature_cols,round_predictions=True,
                                                                   plot_results = True)
        valid_pred_sarimax_table=validation_data.copy()
        valid_pred_sarimax_table["predictions"]=valid_pred_sarimax
        # check accuracy of predictions
        
        mae_sarimax = mean_absolute_error(valid_pred_sarimax_table['actual'],valid_pred_sarimax_table['predictions'])
    
    
    #***********************************************************************************
    # Fit Model

    if True:#model_type=='sarima':
        print("I am in pred step -fit")
        print("model type is")
        print(model_type)
        try:
            sarima_model,pred_sarima= predict_sarima_model(train_set=train_model.loc[:,target_col],
                                       test_set=test_data.loc[:,target_col],
                                                    order=grid_pdq,
                                                    seasonal_order=seasonal_grid_pdq,
                                                            plot_results = True,round_predictions=True)
            
            result = test_data.merge(pred_sarima,left_index=True,right_index=True)
            result.loc[:,'model_type'] = 'sarima'
            return sarima_model
        except:
            pass
        
    elif model_type=='sarimax':
    
        try:
            sarimax_model,pred_sarimax = predict_sarimax_model(x_train=train_model.loc[:,feature_cols],
                                                               y_train=train_model.loc[:,target_col],
                                                               x_test=test_data.loc[:,feature_cols], 
                                                               y_test=test_data.loc[:,target_col],
                                                               order=grid_pdq,seasonal_order=seasonal_grid_pdq,
                                                               feats_to_use=feature_cols,round_predictions=True,
                                                               plot_results = True)
            
            result = test_data.merge(pred_sarimax,left_index=True,right_index=True)
            result.loc[:,'model_type'] = 'sarimax'
        except:
            pass       
    
        
    elif model_type=="ma":
        result = test_data.copy()
        result.loc[:,'predictions'] = round(train_model.loc[:,target_col].tail(4).mean())
        result.loc[:,'model_type'] = 'ma'
    #all_results = pd.concat([all_results,tmp_result],axis=0)
    #***********************************************************************************
    

def prediction_func(data,pdq,seasonal_pdq,test_day,model_type,freq="D"): 
 
    data.iloc[:,0]=pd.to_datetime(data.iloc[:,0])
    data=data.set_index(list(data)[0])
        
    '''
    test_date = pd.to_datetime(test_date)
    test_date_end = test_date+ datetime.timedelta(days=4)
    last_week = (pd.to_datetime(test_date) - datetime.timedelta(days=7))

    train_data = data.loc[:last_week,:]
    test_data = data.loc[test_date:test_date_end,:]
    validation_data=data.loc[last_week:test_date,:]'''
    valid_day=test_day+7
    train_data=data.iloc[:-valid_day]
    test_data = data.iloc[-test_day:]
    validation_data=data.iloc[-valid_day:-test_day]
    # train/test/validation split
    

    if (train_data.shape[0] !=0) and (validation_data.shape[0] !=0) and (test_data.shape[0] !=0):
        
        if model_type=="ma":
            model = prediction_step(train_data,validation_data,test_data,pdq=(1,0,0),seasonal_pdq=(0,0,0,0),model_type='ma')
            print(model)
        else:
            print("I am in pred func")
            print("model type is")
            print(model_type)
            model = prediction_step(train_data,validation_data,test_data,pdq,seasonal_pdq,model_type=model_type)
            print(model)
        '''    
        prediction_dir = "/home/yunus-emre.karatas/time_series_tool/weekly_prediction/predictions/" + str(datetime.datetime.now()).replace("-","_").replace(" ","_").split(".")[0].replace(":","_") 

        Path(prediction_dir).mkdir(parents=True, exist_ok=True)

        prediction_dir = prediction_dir +"/"
            
        
        pred_file_name =prediction_dir+str(test_date)+ "_"+".csv"
        result.to_csv(pred_file_name)'''
        
    else:
        pass
        
    logging.captureWarnings(True)
    warnings.filterwarnings(action='once')

    return model 



    
