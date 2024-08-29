import numpy as np
import pandas as pd
import tensorflow as tf

from prep import FeaturePreProcessing
from LSTM import nn_model
from plot_pred import plot_predictions
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

# Read training stock data files
bp_data = pd.read_excel(r'.\data\training\BP.L_filtered.xlsx')
dge_data = pd.read_excel(r'.\data\training\DGE.L_filtered.xlsx')
gsk_data = pd.read_excel(r'.\data\training\GSK.L_filtered.xlsx')
hsba_data = pd.read_excel(r'.\data\training\HSBA.L_filtered.xlsx')
ulvr_data = pd.read_excel(r'.\data\training\ULVR.L_filtered.xlsx')

# Create a dictionary of training data
training_data = {'bp': bp_data
                 ,'dge': dge_data
                 ,'gsk': gsk_data
                 ,'hsba': hsba_data
                 ,'ulvr': ulvr_data
                }

# Read test stock data files
azn_data = pd.read_excel(r'.\data\testing\AZN.L_filtered.xlsx')
barc_data = pd.read_excel(r'.\data\testing\BARC.L_filtered.xlsx')
rr_data = pd.read_excel(r'.\data\testing\RR.L_filtered.xlsx')
tsco_data = pd.read_excel(r'.\data\testing\TSCO.L_filtered.xlsx')
vod_data = pd.read_excel(r'.\data\testing\VOD.L_filtered.xlsx')

# Create a dictionary of test data
test_data = {'azn': azn_data
             ,'barc': barc_data
             ,'rr': rr_data
             ,'tsco': tsco_data
             ,'vod': vod_data
            }

random_seed = 42
tf.random.set_seed(random_seed)

# Create a feature engineering object
feature_eng = FeaturePreProcessing()

# set the number of lag days to obtain
lag_days = 5

# loop through each training data and train prediction models 
# each model is evaluated on all the test data
for train_name, train in training_data.items():
    print("="*50)
    print(f"Training on {train_name} ...")
    print("="*50)
    train = train.copy()

    # Calculate the log of adj close column
    train['Adj Close'] = np.log(train['Adj Close'])

    # Obtain lag columns and date hierarchies
    train = feature_eng(train, lag_days=lag_days)

    # Replace nan values with 0
    train = train.fillna(0)

    # Remove unwanted columns
    train = train.drop(columns=['Close','Date','year','High','Low','Open','Volume'])

    # Get column names
    cols = train.columns.tolist()

    # Remove month, day, day_of_year and adj close from cols
    feature_cols = set(cols)-set(['month','day','day_of_year','Adj Close'])
    # change feature_cols to list object
    feature_cols = list(feature_cols)

    # scale train input
    train_input_scaler = RobustScaler()
    train[feature_cols] = train_input_scaler.fit_transform(train[feature_cols])

    # scale output(adj close) values
    train_output_scaler = RobustScaler()
    train_output_scaler.fit(train[['Adj Close']])

    train_y = train_output_scaler.transform(train[['Adj Close']])
    train_x = train.drop(columns=['Adj Close'])

    # train test split
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, shuffle=True, test_size=0.2, random_state=42)

    # Train random forest model
    rf_regr = RandomForestRegressor(n_estimators=200, random_state=random_seed, verbose=0)
    # Fit random forest model on 80% split of training data (train_x, train_y)
    rf_regr.fit(train_x, train_y)

    # Evaluate random forest model on 20% of training data split (valx, valy)
    pred = rf_regr.predict(val_x)
    pred = train_output_scaler.inverse_transform(pred.reshape(-1,1))
    pred = np.exp(pred)
    y_true = np.exp(train_output_scaler.inverse_transform(val_y))
    rms_error = root_mean_squared_error(y_true, pred)
    ma_error = mean_absolute_error(y_true, pred)
    print("Inferencing with Random Forest model ...")
    print(f"The root mean squared error on the validation data is {rms_error}.")
    print(f"The mean absolute error on the validation data is {ma_error}.")

    # Train gradient boosting model
    xgb_regr = GradientBoostingRegressor(n_estimators=200, random_state=random_seed, verbose=0)
    # Fit gradient boosting model on 80% split of training data (train_x, train_y)
    xgb_regr.fit(train_x, train_y)

    # Evaluate gradient boosting model on 20% of training data split (valx, valy)
    pred = xgb_regr.predict(val_x)
    pred = train_output_scaler.inverse_transform(pred.reshape(-1,1))
    pred = np.exp(pred)
    y_true = np.exp(train_output_scaler.inverse_transform(val_y))
    rms_error = root_mean_squared_error(y_true, pred)
    ma_error = mean_absolute_error(y_true, pred)
    print("Inferencing with Gradient Boosting model ...")
    print(f"The root mean squared error on the validation data is {rms_error}.")
    print(f"The mean absolute error on the validation data is {ma_error}.")

    # Fit linear regression model on 80% split of training data (train_x, train_y)
    linear_regr = LinearRegression()
    linear_regr.fit(train_x, train_y)

    # Evaluate linear regression on 20% of training data split (valx, valy)
    pred = linear_regr.predict(val_x)
    pred = train_output_scaler.inverse_transform(pred.reshape(-1,1))
    pred = np.exp(pred)
    y_true = np.exp(train_output_scaler.inverse_transform(val_y))
    rms_error = root_mean_squared_error(y_true, pred)
    ma_error = mean_absolute_error(y_true, pred)
    print("Inferencing with Linear Regression model ...")
    print(f"The root mean squared error on the validation data is {rms_error}.")
    print(f"The mean absolute error on the validation data is {ma_error}.")

    # Train LSTM model
    model = nn_model()
    tf.keras.utils.plot_model(model, "model_architecture.png", show_shapes=True)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['MAE'],
    )
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, 
                                    verbose=0, mode='min', start_from_epoch=100, restore_best_weights=True)

    history = model.fit(train_x, train_y, batch_size=64, callbacks=[es], epochs=200, validation_split=0.2, verbose=0)

    # Evaluate LSTM model on validation data (test_x, test_y)
    pred = model.predict(val_x, verbose=0)
    pred = train_output_scaler.inverse_transform(pred.reshape(-1,1))
    pred = np.exp(pred)
    y_true = np.exp(train_output_scaler.inverse_transform(val_y))
    rms_error = root_mean_squared_error(val_y, pred)
    ma_error = mean_absolute_error(y_true, pred)
    print("Inferencing with LSTM model ...")
    print(f"The root mean squared error on the validation data is {rms_error}.")
    print(f"The mean absolute error on the validation data is {ma_error}.")

    # Test models on test data
    print(f"{'*'*50} \nTesting on test datasets\n{'*'*50}")

    for test_name, test in test_data.items():
        print("-"*50)
        print(f"Testing on {test_name}")

        test = test.copy()

        # create prediction data
        pred_df = test[['Date','Adj Close']]
        pred_df.rename(columns={'Adj Close': 'y_true'}, inplace=True)

        test['Adj Close'] = np.log(test['Adj Close'])

        # scale data on test adj close columns
        test_output_scaler = RobustScaler()
        test_output_scaler.fit(test[['Adj Close']])

        test = feature_eng(test, lag_days=lag_days)

        # Replace nan values with 0
        test = test.fillna(0)
        test = test.drop(columns=['Close','Date','year','High','Low','Open','Volume'])

        # scale data on input columns
        test_input_scaler = RobustScaler()
        test[feature_cols] = test_input_scaler.fit_transform(test[feature_cols])

        test_y = np.exp(test['Adj Close'])
        test_x = test.drop(columns=['Adj Close'])

        # Evaluate random forest model on test data (test_x, test_y)
        pred = rf_regr.predict(test_x)
        pred = test_output_scaler.inverse_transform(pred.reshape(-1,1))
        pred = np.exp(pred)
        pred_df['y_pred'] = pred
        plot_predictions(pred_df, train_name, test_name, 'randomForest')
        rms_error = root_mean_squared_error(test_y, pred)
        ma_error = mean_absolute_error(test_y, pred)
        print("Inferencing with Random Forest model ...")
        print(f"The root mean squared error on the test data is {rms_error}.")
        print(f"The mean absolute error on the test data is {ma_error}.")

        # Evaluate gradient boosting model on test data (test_x, test_y)
        pred = xgb_regr.predict(test_x)
        pred = test_output_scaler.inverse_transform(pred.reshape(-1,1))
        pred = np.exp(pred)
        pred_df['y_pred'] = pred
        plot_predictions(pred_df, train_name, test_name, 'XGBoost')
        rms_error = root_mean_squared_error(test_y, pred)
        ma_error = mean_absolute_error(test_y, pred)     
        print("Inferencing with Gradient Boosting model ...")
        print(f"The root mean squared error on the test data is {rms_error}.")
        print(f"The mean absolute error on the test data is {ma_error}.")

        # Evaluate linear regression model on test data (test_x, test_y)
        pred = linear_regr.predict(test_x)
        pred = test_output_scaler.inverse_transform(pred.reshape(-1,1))
        pred = np.exp(pred)
        pred_df['y_pred'] = pred
        plot_predictions(pred_df, train_name, test_name, 'linearRegression')
        rms_error = root_mean_squared_error(test_y, pred)
        ma_error = mean_absolute_error(test_y, pred)
        print("Inferencing with Linear Regression model ...")
        print(f"The root mean squared error on the test data is {rms_error}.")
        print(f"The mean absolute error on the test data is {ma_error}.")

        # Evaluate LSTM model on test data (test_x, test_y)
        pred = model.predict(test_x, verbose=0)
        pred = test_output_scaler.inverse_transform(pred.reshape(-1,1))
        pred = np.exp(pred)
        pred_df['y_pred'] = pred
        plot_predictions(pred_df, train_name, test_name, 'LSTM')
        rms_error = root_mean_squared_error(test_y, pred)
        ma_error = mean_absolute_error(test_y, pred)
        print("Inferencing with LSTM model ...")
        print(f"The root mean squared error on the test data is {rms_error}.")
        print(f"The mean absolute error on the test data is {ma_error}.")