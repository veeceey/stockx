# stockx


CS235
Prediction and Interpretation of Stock Price using Data Mining Techniques
-Vagelis Papalexakis

Prince Choudhary
Department of Computer Science
University of California, Riverside
pchou014@ucr.edu

Varun Chawla
Department of Computer Science
University of California, Riverside
vchaw001@ucr.edu

Shreya Singh
Department of Computer Science
University of California, Riverside
ssing224@ucr.edu

1) Introduction

In this project we intend to assess the performance of various Recurrent Neural Networks to predict the stock price of Apple. RNNs are good for time-series analysis, therefore we look at the performance of Vanilla RNN, Bidirectional LSTM and GRU.

Vanilla RNNs are neural networks that are good at modeling sequence data for prediction. They do the future prediction by a concept called sequential memory. A typical RNN cell has an input layer, hidden layer and output layer. It has a looping mechanism that acts as a highway to allow information to flow from one step to the next. Although it faces short term memory problems. One advantage of Vanilla RNN is that training data is fast, it uses less computational resources because it has less tensor operations to compute.

Bidirectional LSTMs learn the input sequence from both forward and backward directions and concatenate both of those interpretations. These kinds of models work well on Time Series data owing to the fact that unlike competing regular standard neural networks, LSTM not only takes into account feedback connections, but also processes and predicts over an entire sequence of data which is a helpful quality while dealing with Time-Series data like Stock Market Data. 
GRU (Gated Recurrent Unit) attempts to solve the vanishing gradient problem of standard Recurrent Neural Network and can be considered as a variation on the LSTM model, just that it has fewer parameters and lacks an output gate and only has the update and reset gate. It is a good candidate when considering a model for time series analysis. It is supposed to exhibit better performance on certain smaller and less frequent datasets.

2) Related Work

The prediction of stock market price is not a new problem and several methods have been devised by different researchers to predict stock price with better accuracy and lesser error. In recent days there has been a lot of work on stock market prediction from deep learning communities. Adebiyi, Adewumi, and Ayo [1] worked on a multilayer perceptron model. Rather, Agrawal and Sastry [2] worked on mixed versions of statistical models and RNNs.


3) Proposed Method

In the proposed model of Vanilla RNN we make forward passes to make predictions, compare the predictions to ground truth using loss function. The loss function outputs an ‘error value’ which is an estimate of how badly the network is performing. Uses the ‘error value’  for back propagation which calculates gradient for each node in the network. The gradient is the value we use to adjust a network's internal weight, allowing the network to learn. Fig 1 is a depiction of typical RNN cells work flow.



In the creation of Bi-directional LSTM model different parameters were used with the most important one being the historical sequence of days to predict future values. With scaling, test and train data split ratio of 0.2, the model was created with 256 neural networks and trained across varying passes ranging from 50 to 200. As the model trains in every pass it returns loss error which it tries to reduce in every pass until finally converging within the set number of epochs(passes). The model was feeded with Adam optimizer  for stochastic gradient descent to predict adj close values which are the closing values of a stock in a day after all pending settlements related to stocks are done by a company. . The following feature columns of the stock were used ["adjclose", "volume", "open", "high", "low”]. All of these values were fed into the model operating bidirectionally to predict future adjclose value of the stock AAPL. After the model converges to a loss error value iterating through multiple passes, we use prediction and evaluation functions to predict values based on past training and testing of data and also to evaluate the accuracy of the predictions by summing count of positive profits only and dividing them by number of stock trade entries. Huber Loss and Mean Absolute error were calculated for testing the efficiency of the model performance. 

GRU uses two vectors called update gate and reset gate which decide which information is to be passed to the output. The update gate decides how much of the past information must be passed along to the future. The reset gate decides how much of the past information to forget. It is a good candidate when considering a model for time series analysis.

4) Experimental Evaluation
	
For the Vanilla RNN model, we trained the model with the last 1 year of apple stock data  and tested it in the last 45 days. We conducted 5 simulations with epochs set to 200. The results were approximately 90% accurate.

Bidirectional LSTM was trained on Macbook Air with 8 GB RAM and 256 GB SSD and on Google Collaboratory having 12 GB of Ram and 64 GB of disk space. The best results were predicted under this model with a huber loss of  0.0007819203310646117, Mean Absolute Error: 4.043143804308628 and Accuracy score: 0.6424361493123772 converging loss value to 0.00078 in 200 epochs. The model was trained on the last 20 years of data and tested on the last 2 years of data.   The LSTM model predicted Future price of stock after 15 days fromJune 10 to be 128.85$. The current stock price of AAPL at the time of writing the report was 127.35 After the run till Epoch 00200 val_loss did not improve from 0.00078. Future price of the stock after 15 days was predicted to be 128.85$. The predicted price is aligning well with the actual price of the Stock AAPL. As you could see the model predicted 128.85$ and the current market price is 127.35, and is roughly within 1 % of the current stock market price. Observing the current performance of AAPL in the stock market, i don’t think it will go over 130 in the next 15 days. I passively invest in stocks and I feel AAPL is not performing that well on the market past some 6-12 months and Apple is trending between 140 and 116 USD. I feel my predicted price should align with the actual price of the stock.  Figure A was trained on GoogleCollab while was Figure B was trained on Macbook with LookupSteps = 30 i.e. predict the next sequence window from existing sequence window spanning across 30 days, as described in Figure B
Result was trained on Bi-directional LSTM with Window Size of Days to be 25 and Lookup Steps to be 15 with 256 neurons


Figure B- result was trained on Bi-directional LSTM with Window Size of Days to be 5 and Lookup Steps to be 30 with 256 neurons having an Accuracy score: 0.487 and Future price after 30 days to be 114.82$

The GRU model was trained on Apple stock data which was collected using a python package called yahoo_fin which contains functions to scrape stock related data from Yahoo Finance. The data is normalized from 0 to 1 and the NaN values are dropped. The data is split by date into training and test set (80:20  split). In the large data set we consider prices from 1980 to present and in the smaller set we consider from 2017 to present. In constructing the model, we choose window size of 100, 200 number of  units, 2 layers, 40% dropout rate, mean squared error loss, batch size of 50, epochs as 200 and the linear activation function. We observe mean squared error loss, mean squared error and accuracy of the model to see the performance.
On the large dataset, the mean squared error is 0.062, the accuracy is 0.52 and the mean squared error loss is 0.000168.  The future price for the next day is predicted as 124.37$. Below is the graph with actual and predicted values over time.


 On the small dataset, the mean squared error is 27.45, the accuracy is 0.56 and the mean squared error loss is 0.000666.  The future price for the next day is predicted as 126.67$. Below is the graph with actual and predicted values over time.










5) Discussion & Conclusions

The models seem to have varying performance. All of them at least predict the fluctuation of the stock properly as can be seen from the graphs.It might be beneficial to apply the models to minute-level or second-level stock prices given the dynamic nature of stock markets. A daily level stock price has about hundreds of data points. However, at this time we do not have that kind of data and computing resources.

Still we achieve the aim of our project to gauge the performance of various Recurrent Neural Network models, used for time series analysis, in stock market prediction.


6) Contributions

Each member has worked on one model end to end, however, everyone has participated in data gathering, data preprocessing, normalization, feature extraction, and project progress evaluation along with assisting team-mates in cross-module implementation. We had divided our project’s implementation in modules that were developed parallely based on Data Mining specific techniques put into use by each one of us, as below-

	Shreya - GRU implementation.
	Prince - Vanilla RNN implementation.
	Varun -  Bidirectional LSTM


7) References

Lecture slides -Vagelis Papalexakis
[1] Ayodele Ariyo Adebiyi. Aderemi Oluyinka Adewumi. Charles Korede Ayo. "Comparison of ARIMA and Artificial Neural Networks Models for Stock Price Prediction." J. Appl. Math. 2014 1 - 7, 2014. https://doi.org/10.1155/2014/614342
[2] DAC '91: Proceedings of the 28th ACM/IEEE Design Automation ConferenceJune 1991 Pages 452–457https://doi.org/10.1145/127601.127711
https://www.python.org/doc/
https://www.tensorflow.org/
https://medium.com/@apiltamang/unmasking-a-vanilla-rnn-what-lies-beneath-912120f7e56c
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
https://keras.io/api/layers/recurrent_layers/bidirectional/
https://www.w3schools.com/python/numpy/numpy_intro.asp



