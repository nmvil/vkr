# -*- coding: utf-8 -*-

import time
import pandas as p
import numpy as n
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy import sparse
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier

# старт таймера
start_time = time.time()

# функция для печати времени выполнения программы


def printTime(timer, start_time=start_time):
    time = ' ---  '
    sec = (timer - start_time)
    if sec > 60:
        minute = sec // 60
        sec = sec % 60
        if minute > 60:
            hour = minute // 60
            minute = minute % 60
            if hour > 23:
                days = hour // 24
                hour = hour % 24
                time += str(days) + ' days '
            time += str(hour) + ' hour '
        time += str(minute) + ' minutes '
    time += str('{:.3f}'.format(sec)) + ' seconds ---\n'
    print(time)


# Метод опорных векторов с линейным ядром
def SVC(x_train, y_train, x_test, y_test):
    timer = time.time()
    # сам метод
    svc = LinearSVC(C=100, penalty='l1', dual=False)
    svc.fit(x_train, y_train)
    svс_pred_ng = svc.predict(x_test)
    print('\033[01;38;05;70mМетод опорных векторов с линейным ядром\033[0m')
    print('точность: {}\nполнота: {}\nF-мера: {}\nматрица ошибок: \n{}\n\n'.format(precision_score(y_test, svс_pred_ng),
                                                                                   recall_score(y_test, svс_pred_ng),
                                                                                   f1_score(y_test, svс_pred_ng),
                                                                                   confusion_matrix(y_test, svс_pred_ng)))
    printTime(time.time(), timer)


# Наивный Байесовский классификатор
def Bayes(x_train, y_train, x_test, y_test):
    timer = time.time()

    NB = MultinomialNB()
    NB.fit(x_train, y_train)
    NB_pred_ng = NB.predict(x_test)
    print('\033[01;38;05;70mНаивный Байесовский классификатор\033[0m')
    print('точность: {}\nполнота: {}\nF-мера: {}\nматрица ошибок: \n{}\n\n'.format(precision_score(y_test, NB_pred_ng),
                                                                                   recall_score(y_test, NB_pred_ng),
                                                                                   f1_score(y_test, NB_pred_ng),
                                                                                   confusion_matrix(y_test, NB_pred_ng)))
    printTime(time.time(), timer)


# Градиентный бустинг над решающими деревьями
def LGBM(x_train, y_train, x_test, y_test):
    timer = time.time()

    lgbm = LGBMClassifier()
    lgbm.fit(x_train, y_train)
    LB_pred_ng = lgbm.predict(x_test)
    print('\033[01;38;05;70mГрадиентный бустинг над решающими деревьями\033[0m')
    print('точность: {}\nполнота: {}\nF-мера: {}\nматрица ошибок: \n{}\n\n'.format(precision_score(y_test, LB_pred_ng),
                                                                                   recall_score(y_test, LB_pred_ng),
                                                                                   f1_score(y_test, LB_pred_ng),
                                                                                   confusion_matrix(y_test, LB_pred_ng)))
    printTime(time.time(), timer)


# чтение базы сообщений
msg = p.read_csv('./database/Reddit_Data.csv', names=['text', 'mood'], sep=';', encoding='UTF-8', engine='c')
msg = shuffle(msg).reset_index(drop=True)

# разбиение выборки
text = p.DataFrame(msg['text'])
mood = p.DataFrame(msg['mood'])


# извлечение первичных признаков
print('\033[01;38;05;70mИзвлечение первичных признаков\033[0m')

timer = time.time()
# подсчёт слов
text['word_count'] = text['text'].apply(lambda text: len(str(text).split(" ")))
print('\033[01;38;05;70mСлова посчитаны\033[0m')
printTime(time.time(), timer)

timer = time.time()
# средняя длина слова


def avg_word_len(text):
    words = str(text).split()
    length = (sum(len(word) for word in words) / len(words))
    return length


text['average_word_length'] = text['text'].apply(avg_word_len)
print('\033[01;38;05;70mСредняя длина слов посчитана\033[0m')
printTime(time.time(), timer)

timer = time.time()
# убираем "стоп слова"
text['stopwords_count'] = text['text'].apply(lambda text: len([stopword for stopword in str(text).split() if stopword in stopwords.words('english')]))
print('\033[01;38;05;70mУбраны "стоп слова"\033[0m')
printTime(time.time(), timer)

timer = time.time()
# процесс стемминга
stem = SnowballStemmer('english')
text['text'] = text['text'].apply(lambda text: " ".join([stem.stem(word) for word in str(text).split()]))
# удаление из текста смеха вида 'hahahhaha' 'ahahahahha'
text['text'] = text['text'].apply(lambda text: "".join(re.sub(r'\b(a*ha+h[ha]*|o?l+o+l+[ol]*)\b', r"haha", str(text))))
print('\033[01;38;05;70mЗавершён процесс стемминга\033[0m')
printTime(time.time(), timer)

timer = time.time()
# векторизация текста
text_matrix = sparse.csr_matrix(text.drop('text', axis=1))

# векторизация с помощью N-грамм
ngramm = CountVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0001)
x_ngram = ngramm.fit_transform(text['text'].values)
ngram_names = ngramm.get_feature_names()
x_ngram = hstack([text_matrix, x_ngram])
x_train, x_test, y_train, y_test = train_test_split(x_ngram, mood.values, test_size=0.3, shuffle=True)
print('\033[01;38;05;70mЗавершён процесс векторизации с помощью N-грамм\033[0m')
printTime(time.time(), timer)


timer = time.time()
# Метод опорных векторов с линейным ядром
# сетка
grid = {'C': n.power(10.0, n.arange(-3, 4))}
cv = KFold(n_splits=5, shuffle=True)
svc = LinearSVC(penalty='l1', dual=False)
gs = GridSearchCV(svc, grid, scoring='precision', cv=cv)
gs.fit(x_ngram, mood.values.ravel())
mean_svm_score = gs.cv_results_['mean_test_score']
C_values = gs.cv_results_['params']
for i in range(len(mean_svm_score)):
    print('\033[01;38;05;178m {} {}\033[0m'.format(mean_svm_score[i], C_values[i]))

print('\033[01;38;05;70mСетка для метода опорных векторов с линейным ядром\033[0m')
printTime(time.time(), timer)

# Метод опорных векторов с линейным ядром
SVC(x_train, y_train, x_test, y_test)
# Наивный Байесовский классификатор
Bayes(x_train, y_train, x_test, y_test)
# Градиентный бустинг над решающими деревьями
LGBM(x_train, y_train, x_test, y_test)



# TF-IDF



print('\033[01;38;05;70mTF-IDF\033[0e')
timer = time.time()
# векторизация TF_IDF
delta_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0001)
delta_tfidf = delta_vectorizer.fit_transform(text['text'].values, list(mood))
delta_names = delta_vectorizer.get_feature_names()
delta_tfidf = hstack([text_matrix, delta_tfidf])
x_train, x_test, y_train, y_test = train_test_split(delta_tfidf, mood.values, test_size=0.3, shuffle=True)
print('\033[01;38;05;70mВекторизация TF-IDF завершена\033[0m')
printTime(time.time(), timer)

# Метод опорных векторов с линейным ядром (TF-IDF)
SVC(x_train, y_train, x_test, y_test)
# Наивный Байесовский классификатор (TF-IDF)
Bayes(x_train, y_train, x_test, y_test)
# Градиентный бустинг над решающими деревьями (TF-IDF)
LGBM(x_train, y_train, x_test, y_test)



# Объединённый признак


print('\033[01;38;05;70mОбъединённый признак\033[0m')
timer = time.time()

All_features = hstack([x_ngram, delta_tfidf])
x_train, x_test, y_train, y_test = train_test_split(All_features, mood.values, test_size=0.3, shuffle=True)
printTime(time.time(), timer)


# Метод опорных векторов с линейным ядром (объединённый признак)
SVC(x_train, y_train, x_test, y_test)
# Наивный Байесовский классификатор (объединённый признак)
Bayes(x_train, y_train, x_test, y_test)
# Градиентный бустинг над решающими деревьями (объединённый признак)
LGBM(x_train, y_train, x_test, y_test)


# печать времени работы
printTime(time.time())
