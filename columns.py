from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os, joblib, time

dump = 'forest.dump'

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
categorial_columns = set_names([1, 3, 5, 8, 9, 11, 14, 15, 16, 17, 18, 20, 24, 26, 28])
digit_columns = set_names([4, 6, 7, 12, 13, 19, 27])
binary_columns = set_names([])
delete_columns = set_names([21, 22])

# Удаляем ненужные столбцы
x = x.drop(delete_columns, axis=1)
print('Выполнение One Hot Encoding')
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(x[categorial_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, 
                          columns=encoder.get_feature_names_out(categorial_columns), 
                          dtype=int)
x = pd.concat([x, one_hot_df], axis=1)
x = x.drop(categorial_columns, axis=1)

print('Заполнение пустых элементов')
x = x.fillna(10)

print('Стандратизация данных')
scaler = StandardScaler()
x[digit_columns] = scaler.fit_transform(x[digit_columns])

def correl(X_train, thresh) -> pd.Series:
    cor = X_train.corr()
    c1 = cor.stack().sort_values(ascending=False).drop_duplicates()
    all_cor = c1[c1.values != 1]
    return all_cor[abs(all_cor) > thresh]

print('Расчет кореляций')
correlations_data = x.corr()

print('Расчет корреляции для столбцов')
columns, correlations = [], []
for col in x.columns:
    columns.append(col)
    correlations.append(stats.pointbiserialr(x[col], Y)[0])

print('Вывод тепловой карты для корреляции всех признаков')
plt.title('Вывод тепловой карты для корреляции всех признаков')
sns.heatmap(correlations_data, 
            cmap='coolwarm', 
            square=True, 
            fmt='.2f')
plt.show() ### ###

print('Выборка наиболее коррелиуермых признаков')
best = correl(x, 0.2)
x = x[pd.Index(best.index.get_level_values(0).to_list() + best.index.get_level_values(0).to_list()).unique()]

print('Расчет кореляций')
correlations_data = x.corr()

print('Тепловая карта для кореллируемых значений')
plt.title('Тепловая карта для кореллируемых значений')
sns.heatmap(correlations_data, 
            # annot=True, 
            cmap='coolwarm', 
            square=True, 
            # linewidths=0.5,
            fmt='.2f')
plt.show() ### ###

print('Обучение модели классификатора случайного леса для определения важности коэфициентов')
rf = RandomForestClassifier(n_estimators=100, random_state=229)
rf.fit(x, Y)
importances = pd.Series(rf.feature_importances_, index=x.columns)

print('Самые важные характеристики')
print(importances.sort_values(ascending=False).head(20))
print('Самые неважные характеристики')
print(importances.sort_values(ascending=True).head(20))

print('Выборка только важных признаков')
x = x[importances.sort_values(ascending=False).head(20).index]

print('Тепловая карта после выборки только важных признаков')
plt.title('Тепловая карта после выборки только важных признаков')
correlations_data = x.corr()
sns.heatmap(correlations_data, 
            cmap='coolwarm', 
            square=True, 
            fmt='.2f')
plt.show() ### ###

print('Создание тестовых данных')
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, 
                                                    test_size = 0.01, 
                                                    random_state = 456)
print(x.shape, Y.shape, x_train.shape, x_test.shape, Y_train.shape, Y_test.shape)

logreg = None

if os.path.exists(dump):
    # Выгрузка сохраненной модели
    print('Модель уже обучена, выгружаем')
    logreg = joblib.load(dump)

else:
    # Создание и обучение модели
    start_time = time.time()
    print('Обучаем модель')

    logreg = RandomForestClassifier(n_estimators=100, random_state=229)
    logreg.fit(x_train, Y_train)

    print('\nВремя обучения: ', time.time() - start_time)

    # Сохранение дампа объекта модели
    joblib.dump(logreg, dump)

Y_pred_train = logreg.predict(x_train)
sns.heatmap(confusion_matrix(Y_train, Y_pred_train), annot = True, fmt='g')
print('\nTrain data:')
print('Recall (train) =', recall_score(Y_train, Y_pred_train))
print('Precision (train) =', precision_score(Y_train, Y_pred_train))
print('F1 (train) =', f1_score(Y_train, Y_pred_train))
print('Accuracy (train) =', accuracy_score(Y_train, Y_pred_train))
print('AUC (train) =', roc_auc_score(Y_train, Y_pred_train))
plt.show() ###

Y_pred_test = logreg.predict(x_test)
sns.heatmap(confusion_matrix(Y_test, Y_pred_test), annot = True, fmt='g')
print('\nTest data:')
print('Recall (test) =', recall_score(Y_test, Y_pred_test))
print('Precision (test) =', precision_score(Y_test, Y_pred_test))
print('F1 (test) =', f1_score(Y_test, Y_pred_test))
print('Accuracy (test) =', accuracy_score(Y_test, Y_pred_test))
print('AUC (test) =', roc_auc_score(Y_test, Y_pred_test))
plt.show() ###
