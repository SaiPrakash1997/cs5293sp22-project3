import argparse
import pandas as pd
import numpy as np
import argparse
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


def predictor(args):
    print("args:", args)
    print("args.text :", args.text)
    passed_sentence = args.text
    dataFrame = read_tsv()
    names, temp_redacted_sentences = extract_sentences(dataFrame)
    _sentences = normalize_sentences(temp_redacted_sentences)
    model_definition(_sentences, names, passed_sentence)


def read_tsv():
    dataFrame = pd.read_csv('unredactor.tsv', sep='\t', on_bad_lines='skip')
    dataFrame.rename(
        columns={"I couldn't image ██████████████ in a serious role, but his performance truly": 'redacted_sentences',
                 'ashton kutcher': 'actor_name', 'training': 'data_type', 'cegme': 'person'}, inplace=True)
    del dataFrame['person']
    dataFrame = dataFrame.dropna(how='any', axis=0)
    return dataFrame


def extract_sentences(dataFrame):
    temp_redacted_sentences = []
    names = []
    df_redacted_sentences = []
    for i in range(0, len(dataFrame), 1):
        df_redacted_sentences.append(dict(dataFrame.iloc[i]))
    for values in df_redacted_sentences:
        temp_data_type = values['data_type']
        if temp_data_type == 'training' or temp_data_type == 'validation':
            temp_redacted_sentences.append(values['redacted_sentences'])
            names.append(values['actor_name'])
    return names, temp_redacted_sentences


def normalize_sentences(temp_redacted_sentences):
    _sentences = []
    lemma = WordNetLemmatizer()
    stop_words = list(stopwords.words('english'))
    stop_words.append("'ll")
    stop_words.append("!")
    stop_words.append(",")
    stop_words.append(".")
    stop_words.append("i.e")
    stop_words.append("(")
    stop_words.append(")")
    stop_words.append(":")
    stop_words.append("'s")
    stop_words.append("n't")
    stop_words.append("``")
    stop_words.append("''")
    stop_words.append("?")
    stop_words.append("...")
    stop_words.append("--")
    stop_words.append("'")
    stop_words.append("`")
    for sentences in temp_redacted_sentences:
        _words = word_tokenize(sentences)
        temp = []
        for word in _words:
            if not (word.lower() in stop_words):
                word = word.lower()
                words_lemma = lemma.lemmatize(word)
                temp.append(words_lemma)
        _sentences.append(' '.join(temp))
    return _sentences


def model_definition(_sentences, names, passed_sentence):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    frequency = vec.fit_transform(_sentences).toarray()
    le = LabelEncoder()
    names_labels = le.fit_transform(names)
    X_train, X_test, y_train, y_test = train_test_split(frequency, names_labels, test_size=0.2, shuffle=True)
    clf = MLPClassifier(hidden_layer_sizes=(100))
    clf.fit(X_train, y_train)
    clf_model_prediction = clf.predict(X_test)
    print(f"accuracy score for the model: {clf.score(X_test, y_test)} \n ")
    print(f"precision score for the model : {precision_score(y_test, clf_model_prediction, average='macro')} \n ")
    print(f"recall score for the model: {recall_score(y_test, clf_model_prediction, average='macro')} \n ")
    print(f"f1_score for the model: {f1_score(y_test, clf_model_prediction, average='macro')} \n ")
    userPref = np.asarray(normalize_sentences(passed_sentence), dtype=object)
    frequency = vec.transform(userPref)
    y_prediction = clf.predict(frequency)
    name_prediction = le.inverse_transform(y_prediction)[0]
    print(f" Prediction: {name_prediction}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Pass the redacted text sentence", required=True)
    args = parser.parse_args()
    predictor(args)