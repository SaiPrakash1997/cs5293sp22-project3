import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from sklearn.pipeline import make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


def predictor():
    url = r"https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    dataFrame = read_tsv(url)
    names, test_names, temp_redacted_sentences, temp_test_redacted_sentences = extract_sentences(dataFrame)
    _sentences, _test_sentences = normalize_sentences(temp_redacted_sentences, temp_test_redacted_sentences)
    accuracy_score, precisionScore, recallScore, f1Score = model_definition(names, test_names, _sentences, _test_sentences)
    print("******** Output ********")
    print(f"  1. accuracy score for the model: {accuracy_score}")
    print(f"  2. precision score for the model : {precisionScore}")
    print(f"  3. recall score for the model: {recallScore}")
    print(f"  4. f1_score for the model: {f1Score}")


def read_tsv(url):
    dataFrame = pd.read_csv(url, sep='\t', on_bad_lines='skip')
    dataFrame.columns = ['person', 'data_type', 'actor_name', 'redacted_sentences']
    del dataFrame['person']
    dataFrame = dataFrame.dropna(how='any', axis=0)
    le = LabelEncoder()
    dataFrame['names_label'] = le.fit_transform(dataFrame['actor_name'])
    print("Extracted latest unredactor.tsv file and processed it!")
    return dataFrame


def extract_sentences(dataFrame):
    temp_test_redacted_sentences = []
    temp_redacted_sentences = []
    names = []
    test_names = []
    df_redacted_sentences = []
    for i in range(0, len(dataFrame), 1):
        df_redacted_sentences.append(dict(dataFrame.iloc[i]))
    for values in df_redacted_sentences:
        temp_data_type = values['data_type']
        if temp_data_type == 'training' or temp_data_type == 'validation':
            temp_redacted_sentences.append(values['redacted_sentences'])
            names.append(values['names_label'])
        elif temp_data_type == 'testing':
            temp_test_redacted_sentences.append(values['redacted_sentences'])
            test_names.append(values['names_label'])
    return names, test_names, temp_redacted_sentences, temp_test_redacted_sentences


def normalize_sentences(temp_redacted_sentences, temp_test_redacted_sentences):
    _sentences = []
    _test_sentences = []
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
            unicode_str = str(word.encode("unicode_escape"))
            if 'u2588' in unicode_str:
                continue
            if not (word.lower() in stop_words):
                word = word.lower()
                words_lemma = lemma.lemmatize(word)
                temp.append(words_lemma)
                temp_str = ' '.join(temp)
                temp_str = temp_str.strip()
        if len(temp_str) != 0:
            _sentences.append(temp_str)
    for sentences in temp_test_redacted_sentences:
        _words = word_tokenize(sentences)
        temp = []
        for word in _words:
            unicode_str = str((word).encode("unicode_escape"))
            if 'u2588' in unicode_str:
                continue
            if not (word.lower() in stop_words):
                word = word.lower()
                words_lemma = lemma.lemmatize(word)
                temp.append(words_lemma)
                temp_str = ' '.join(temp)
                temp_str = temp_str.strip()
        if len(temp_str) != 0:
            _test_sentences.append(temp_str)
    print("Normalization is applied on data!")
    return _sentences, _test_sentences


def model_definition(names, test_names, _sentences, _test_sentences):
    union = make_union(CountVectorizer(), TfidfVectorizer(ngram_range=(1, 2), min_df=2))
    X = union.fit_transform(_sentences).toarray()
    X_test = union.transform(_test_sentences).toarray()
    union_model = RandomForestClassifier(n_estimators=100)
    union_model.fit(X, names)
    union_model_pred = union_model.predict(X_test)
    accuracy_score = union_model.score(X_test, test_names)
    precisionScore = precision_score(test_names, union_model_pred, average='macro')
    recallScore = recall_score(test_names, union_model_pred, average='macro')
    f1Score = f1_score(test_names, union_model_pred, average='macro')
    return accuracy_score, precisionScore, recallScore, f1Score


if __name__ == '__main__':
    predictor()