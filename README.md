SMS Spam Classifier

The SMS Spam Classifier is a Python script that utilizes machine learning to distinguish between spam (unwanted or malicious) and legitimate (ham) messages in a collection of SMS data. The script employs a popular machine learning algorithm called Naive Bayes, specifically the Multinomial Naive Bayes classifier, to make predictions based on the content of SMS messages.

How it Works:

Data Loading:
The script starts by loading a dataset from a zip file containing SMS data. It extracts the content of the zip file and reads the CSV file ('SMSSpamCollection') using the Pandas library.

Data Preprocessing:
The dataset is preprocessed to map the 'ham' and 'spam' labels to numeric values (0 and 1) for model training.

Training and Testing:
The dataset is split into training and testing sets using the train_test_split function from scikit-learn. This division allows the model to learn patterns from the training set and evaluate its performance on unseen data.

Text Vectorization:
The content of SMS messages is converted into numerical vectors using the CountVectorizer from scikit-learn. This process transforms the text data into a format suitable for machine learning algorithms.

Model Training:
The Multinomial Naive Bayes classifier is trained using the training set to learn the relationships between the features (vectorized messages) and labels (spam or ham).

Prediction:
The trained model is then used to predict whether new SMS messages are spam or ham.

Evaluation:
The script evaluates the model's performance on the test set, providing key metrics such as accuracy, confusion matrix, and a detailed classification report.

Usage:
Users can replace the path to the zip file with their own dataset for SMS messages.
The script can be extended to predict whether new SMS messages are spam or not.
This SMS Spam Classifier serves as a practical tool for automatically identifying and filtering out unwanted or potentially harmful SMS messages, enhancing user experience and security
