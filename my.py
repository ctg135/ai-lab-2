'''

_4 - год выпуска машины. Категориальный или числовой признак? - пока категориальный

числовые признаки (стандартизация):
_6, _7, _9, _12, _13, _14, _15, _16, _17, _18, _19, _27

категориальные признаки (one hot encoding):
_1, _4 (62 уникальных), _5, _20, _21 (192 уникальных), _22 (1475 уникальных), _28

бинарные признаки (чем заполнить пустые?):
_2, _3, _8, _10, _11, _23, _24, _25, _26

target - результирующая колонка


'''


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os, joblib, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


dump = 'model.dump'

print('Открытие .csv')
data = pd.read_csv(r'insclass_train.csv')

def set_names(names: list) -> None:
    for i in range(len(names)):
        names[i] = f'variable_{names[i]}'
    return names

# Сохранение результирующей выборки
Y = data['target']
x = data.drop(['target'], axis=1)

# Обработка входных данных

# Создаем группы столбцов и даем корректные названия
categorial_columns = set_names([1, 4, 5, 7, 20, 21, 24, 22, 28])

digit_columns = set_names([6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 27])

binary_columns = set_names([2, 3, 8, 10, 11, 23, 25, 26])

delete_columns = set_names([])

# Удаляем ненужные столбцы
x = x.drop(delete_columns, axis=1)

# One hot encoding
print('Выполнение One Hot Encoding')
# Создание экземпляра энкодера
encoder = OneHotEncoder(sparse_output=False)
# Полученние значений после one hot encoding
one_hot_encoded = encoder.fit_transform(x[categorial_columns])
# Преобразование полученных значений в массив типа int
one_hot_df = pd.DataFrame(one_hot_encoded, 
                          columns=encoder.get_feature_names_out(categorial_columns), 
                          dtype=int)   
# Конкатенаця полученных значений в результате one hot encoding
x = pd.concat([x, one_hot_df], axis=1)
# Удаляем оставшиеся категориальные признаки
x = x.drop(categorial_columns, axis=1)

# Заолняем пустые значения
print('Заполнение пустых элементов')
x = x.fillna(10)

# Стандартизация числовых полей
print('Стандратизация данных')
for col in digit_columns:
    x[col] = (x[col] - x[col].mean()) / x[col].std()

# Создание тестовых данных и обучение модели

x = x.drop(['variable_6', 'variable_11', 'variable_12', 'variable_13',
       'variable_14', 'variable_16', 'variable_17', 'variable_18',
       'variable_19', 'variable_24', 'variable_26', 'variable_27'], axis=1)



print(x.head())

print('Создание тестовых данных')
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, 
                                                    test_size = 0.35, 
                                                    random_state = 229)
print(x.shape, Y.shape, x_train.shape, x_test.shape, Y_train.shape, Y_test.shape)

# print(x.head)

# rf = RandomForestClassifier(n_estimators=100, random_state=229)
# rf.fit(x_train, Y_train)

# importances = pd.Series(rf.feature_importances_, index=x.columns)
# X_selected = x[importances[importances > 0.01].index]

# print(X_selected)
# print(X_selected.describe())
# print(X_selected.columns)

logreg = None

if os.path.exists(dump):
    # Выгрузка сохраненной модели
    print('Модель уже обучена, выгружаем')
    logreg = joblib.load(dump)

else:
    # Создание и обучение модели
    start_time = time.time()
    print('Обучаем модель')

    logreg = LogisticRegression(max_iter = 2000000, )
    logreg.fit(x_train, Y_train)

    print('\nВремя обучения: ', time.time() - start_time)

    # Сохранение дампа объекта модели
    joblib.dump(logreg, dump)

# Вывод информации о данных

Y_pred_proba = logreg.predict_proba(x_train)
Y_pred_train = logreg.predict(x_train)
sns.heatmap(confusion_matrix(Y_train, Y_pred_train), annot = True, fmt='g')
print('\nTrain data:')
print('Recall (train) =', recall_score(Y_train, Y_pred_train))
print('Precision (train) =', precision_score(Y_train, Y_pred_train))
print('F1 (train) =', f1_score(Y_train, Y_pred_train))
print('Accuracy (train) =', accuracy_score(Y_train, Y_pred_train))
print('AUC (train) =', roc_auc_score(Y_train, Y_pred_train))
plt.show()

Y_pred_test = logreg.predict(x_test)
sns.heatmap(confusion_matrix(Y_test, Y_pred_test), annot = True, fmt='g')
print('\nTest data:')
print('Recall (test) =', recall_score(Y_test, Y_pred_test))
print('Precision (test) =', precision_score(Y_test, Y_pred_test))
print('F1 (test) =', f1_score(Y_test, Y_pred_test))
print('Accuracy (test) =', accuracy_score(Y_test, Y_pred_test))
print('AUC (test) =', roc_auc_score(Y_test, Y_pred_test))
plt.show()

# plt.scatter(x, Y)
# Y_pred = logreg.predict(x)
# Y_proba = logreg.predict_proba(x)
# plt.scatter(x, Y_pred)
# plt.scatter(x, Y_proba[:, 1])
# plt.show()


# Получение вероятностей
y_scores = logreg.predict_proba(x_train)[:, 1]

# Определение порога
precision, recall, thresholds = precision_recall_curve(Y_train, y_scores)

# Выбор порога, который увеличивает Recall
optimal_threshold = thresholds[np.argmax(recall >= 0.08976418144059832)]

# Применение оптимального порога
y_pred = (y_scores >= optimal_threshold).astype(float)


sns.heatmap(confusion_matrix(Y_train, y_pred), annot = True, fmt='g')
print('\nTrain data:')
print('Recall (tresh) =', recall_score(Y_train, y_pred))
print('Precision (tresh) =', precision_score(Y_train, y_pred))
print('F1 (tresh) =', f1_score(Y_train, y_pred))
print('Accuracy (train) =', accuracy_score(Y_train, y_pred))
print('AUC (train) =', roc_auc_score(Y_train, y_pred))
plt.show()
