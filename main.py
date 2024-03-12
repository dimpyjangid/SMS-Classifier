from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

zip_file_path = "../../Downloads/sms+spam+collection.zip"

with ZipFile(zip_file_path, 'r') as zip_file:
    # Read CSV content using Pandas directly from the zip file
    with zip_file.open('SMSSpamCollection', 'r') as csv_file:
        df = pd.read_csv(csv_file, sep='\t', names=['label', 'message'])
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

y_pred = classifier.predict(X_test_vectorized)
for message, prediction in zip(X_test, y_pred):
    result = "Spam" if prediction == 1 else "Not Spam (Ham)"
    print(f'Message: {message}\nPrediction: {result}\n')

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


print(f'Accuracy: {accuracy:.2%}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')