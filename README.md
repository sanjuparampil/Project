# Stock Market Prediction
This project uses machine learning techniques to predict stock market prices for a 10 British companies. It uses historical data to predict future prices. This type of technical analysis is called momentum based trading strategy.

# Datasets Used
- Training sets (2014-2024) - BP PLC (BP.L), Diageo (DGE.L), GlaxoSmithKline PLC (GSK.L), HSBC Holdings (HSBA.L), Unilever PLC (ULVR.L)
- Test set (01/01/2024-30/03/2024) - Barclays (BARC.L), AstraZeneca (AZN.L), Rolls-Royce Holdings PLC (RR.L), Tesco PLC (TSCO.L), Vodafone Group PLC (VOD.L)

## Models
The machine learning models used for this project are:
- Linear regression
- Random Forest
- Gradient Boosting
- Long short-term memory (LSTM) neural network

## Evaluation metrics
- Root mean squared error (RMSE)
- Mean Absolute Error (MAE)

## Results
It was evident that Random Forest has the lowest errors on comparison. Then prediction for the next 3 months was done using this model. Out of 5 testing stocks, the model was able to predict 4 stocks well i.e. BARC.L, RR.L, TSCO.L and VOD.L whereas that of AZN.L was not accurate. 
