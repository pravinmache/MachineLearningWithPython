

```python
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
```


```python
train_df = pd.read_csv('labeledTrainData.tsv', delimiter='\t')
```


```python
test_df = pd.read_csv('testData.tsv', delimiter='\t')
```


```python
STOPWORDS = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
```


```python
def cleanup_review(raw_review):
    '''
        1. remove html tags
        2. remove numbers and other punctuation
        3. convert to lower case remove stopwords
        4. remove stopwords
        
    '''
    raw_review = BeautifulSoup(raw_review,'lxml').get_text()
    raw_review = re.sub("[^A-Za-z]", " ", raw_review)
    word_list = raw_review.lower().split()
    word_list = [word for word in word_list if not word in STOPWORDS]
    word_list = [lemmatizer.lemmatize(word) for word in word_list]
    return " ".join(word_list)
```


```python
cleanup_review("10 spoons of sugar")
```




    'spoon sugar'




```python
cleaned_train_review = []
for review in train_df['review']:
    cleaned_train_review.append(cleanup_review(review))
```


```python
cleaned_test_review =[]
```


```python
cleaned_test_review =[] 
for review in test_df['review']:
    cleaned_test_review.append(cleanup_review(review))
```


```python
len(cleaned_train_review[0])
```




    1404




```python
vectorizer  = CountVectorizer(max_features=6000,
                             stop_words=None)
```


```python
train_features = vectorizer.fit_transform(cleaned_train_review)
```


```python
X_train = pd.DataFrame(train_features.toarray())
y_train = train_df.sentiment
```


```python
test_features = vectorizer.transform(cleaned_test_review)
X_test = pd.DataFrame(test_features.toarray())
```


```python
from sklearn.linear_model import LogisticRegression
```


```python
classifier = LogisticRegression(solver='liblinear', C=0.05)
```


```python
classifier.fit(X_train,y_train)
```




    LogisticRegression(C=0.05, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
              tol=0.0001, verbose=0, warm_start=False)




```python
from sklearn.metrics import accuracy_score
print("training set accuracy :", accuracy_score(y_train, classifier.predict(X_train)))
```

    training set accuracy : 0.93372
    


```python
y_final = classifier.predict(X_test)
```


```python
submission =pd.DataFrame({'id': test_df['id'], 'sentiment':y_final})
```


```python
submission.to_csv('submission.csv', index=False)
```


```python

```
