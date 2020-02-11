import pickle
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

con = sqlite3.connect('db.sqlite')
cur = con.cursor()
cur.execute('select text, class_label from data')
data = cur.fetchall()
con.close()
assert len(data) > 0

X, y = (list(t) for t in zip(*data))
assert len(data) == len(X) == len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Pipeline([
    ('countvect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('sgd', SGDClassifier())
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')

with open('model.pkl', 'wb') as pf:
    pickle.dump(model, pf)
