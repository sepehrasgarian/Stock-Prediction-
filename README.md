# Stock-Prediction-

In this project, we aim to enhance the prediction of stock market movements using sentiment analysis and deep learning.
We divide the effort in this project into four phases. In the first part, we aim to find as much textual data in tweets, comments, etc., as possible. We then process, transform and structure this data so that our models can be trained on it. During the second phase,  pre-trained language models are used to generate sentence-level embeddings for each of the samples in our dataset and save these embeddings on disk. In the third part, a capable classifier is trained to take in the embeddings and predict sentiments. We also aggregate the predicted sentiments to generate a single number indicating how positive the sentiment has been for that stock on that particular day. We save these predicted and aggregated sentiments for each stock symbol and day on disk.
Finally, in the fourth phase, we extract price fluctuations for each stock symbol in each day and compute technical features. Appending the new technical features to the sentiment predictions, we then find the best features and train various hybrid deep learning models to take in these features for a window size before the current day and predict the stock price movement for the next day. 
In this project, we have tested our approaches to a total of 24 Nasdaq stocks. Moreover, the results and methodology are available in the report section.

Here is a youtube link for this project: https://youtu.be/D6BLZUh3QHY

This Projct is developed by: Sepehr Asgarian and Rouzbeh MeshkinNejad


