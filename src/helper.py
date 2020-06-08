import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import contractions
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from xgboost import XGBRegressor
from scipy.sparse import hstack
from scipy.sparse import coo_matrix


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def words_cleaned(data, stemmer):
    '''
    return cleaned reivews 
    '''
    #stopwords list
    stop_list = stopwords.words('english')    
    # creating a list of words that should not be included in stopwords
    not_stop = ["not","don't","aren't","couldn't","didn't","doesn't","hadn't","hasn't","haven't",
               "isn't","mightn't","needn't","shan't","shouldn't","wasn't","weren't","won't","wouldn't","nor", "no"]   
    for word in not_stop:
        stop_list.remove(word)
        
    cleaned = []
    # remove html
    data = data.replace('"', '')
    soup = BeautifulSoup (data, 'html.parser')
    stripped_text = soup.get_text(separator = ' ')
    # remove contractions
    stripped_text = contractions.fix(stripped_text)
    # tokenize 
    tokens = word_tokenize(stripped_text)
    for word in tokens:
        word = re.sub('[^a-zA-z]','', word)
        word = word.lower()

        if word not in set(stop_list) and len(word)>1:
            
            word = stemmer.stem(word)
            cleaned.append(word)

    return (' '.join(cleaned))

def tfidf_vect(train_data, test_data, max_features=20000):
    tfidf = TfidfVectorizer(analyzer = 'word',
                       tokenizer = None,
                       preprocessor = None,
                       stop_words = None,
                       ngram_range=(1,2),
                       max_features=max_features)
    train_tfidf = tfidf.fit_transform(train_data['review_clean'])
    test_tfidf = tfidf.transform(test_data['review_clean'])
    
    return train_tfidf, test_tfidf

def features_regression(df):
    '''
    clean review: lowercase, stemming(porter), remove stopwrods, remove numbers & punctuation 
    added features: review length, sentiment score(using vader), get_dummy on 'condition', # of unique words on cleaned review 
    '''
    df['review_clean'] = df['review'].apply(lambda x : words_cleaned(x, stemmer))
    df['review_len'] = df['review'].apply(lambda x: len(x.split(' ')))
    df['count_unique_word']=df["review_clean"].apply(lambda x: len(set(str(x).split())))
    #df = pd.get_dummies(df, columns=['condition'], drop_first=True)
    
    #sentiment score
    sid = SentimentIntensityAnalyzer()
    df["sentiment_scores"] = df["review"].apply(lambda x: sid.polarity_scores(x))
    df = pd.concat([df.drop(['sentiment_scores'], axis=1), df['sentiment_scores'].apply(pd.Series)], axis=1)
    
    df.drop(columns=['Unnamed: 0','drugName','review','date','neg','neu','pos','condition'], inplace=True)
    
    return df

def features_classification(df):
    '''
    clean review: lowercase, stemming(porter), remove stopwrods, remove numbers & punctuation 
    added features: review length, sentiment score(using vader), get_dummy on 'condition', # of unique words on cleaned review 
    '''
    df['review_clean'] = df['review'].apply(lambda x : words_cleaned(x, stemmer))
    df['review_len'] = df['review'].apply(lambda x: len(x.split(' ')))
    df['count_unique_word']=df["review_clean"].apply(lambda x: len(set(str(x).split())))
    df['useful_class'] = df['usefulCount'].apply(lambda x: 0 if x < 10 else(1 if 10<=x<60 else 2))
    
    #sentiment score
    sid = SentimentIntensityAnalyzer()
    df["sentiment_scores"] = df["review"].apply(lambda x: sid.polarity_scores(x))
    df = pd.concat([df.drop(['sentiment_scores'], axis=1), df['sentiment_scores'].apply(pd.Series)], axis=1)
    
    df.drop(columns=['Unnamed: 0','drugName','review','date','neg','neu','pos','condition','usefulCount'], inplace=True)
    
    return df


def split(df):
    # getting X and y values 
    X = df.drop(columns='usefulCount')
    y = df['usefulCount']
    # split into train & validation/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    return X_train, X_test, y_train, y_test

def split_classification(df):
    # getting X and y values 
    X = df.drop(columns='useful_class')
    y = df['useful_class']
    # split into train & validation/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    return X_train, X_test, y_train, y_test

def drop_na(df):
    df = df.dropna(how='any', axis=0)
    df = df[df['condition'].str.contains('</span>')==False]
    return df

def data_cleanup_for_predict (df_train, df_test):
    df_train = drop_na(df_train)
    df_train = features_regression(df_train)
    
    X_train, X_test, y_train, y_test = split(df_train)
    
    tfidf = TfidfVectorizer(analyzer = 'word',
                        tokenizer = None,
                        preprocessor = None,
                        stop_words = None,
                        ngram_range=(1,2),
                        max_features=20000)
    X_train_tfidf = tfidf.fit_transform(X_train['review_clean'])
    
    
    df_test = features_regression(df_test)
    X = df_test.drop(columns='usefulCount')
    y = df_test['usefulCount']

    X_tfidf = tfidf.transform(X['review_clean'])
    X_test_stacked = hstack([X_tfidf, coo_matrix(X[['rating','review_len','count_unique_word','compound']])])

    return y, X_test_stacked


    
    