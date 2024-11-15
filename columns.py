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
from scipy import stats

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
categorial_columns = set_names([1, 8, 11, 14, 17, 20, 24, 26, 28])

digit_columns = set_names([ 4, 6, 7, 9, 12, 13,  15, 16, 18, 19, 27])

binary_columns = set_names([])

delete_columns = set_names([3, 5, 21, 22])

# Удаляем ненужные столбцы
x = x.drop(delete_columns, axis=1)
# x = x.dropna(thresh=0.3*len(x), axis=1)
# print(x.shape)
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

# print('Заполнение пустых элементов')
# x = x.fillna(10)

print(x)

# Стандартизация числовых полей
print('Стандратизация данных')
print(x.columns)
for col in digit_columns: 
    x[col] = (x[col] - x[col].mean()) / x[col].std()


print(x.shape)
print('Расчет кореляций')
correlations_data = x.corr()
print('Вывод тепловой карты')
sns.heatmap(correlations_data, 
            # annot=True, 
            cmap='coolwarm', 
            square=True, 
            # linewidths=0.5,
            fmt='.2f')
plt.show()



# Заолняем пустые значения
print('Заполнение пустых элементов')
x = x.fillna(10)

# Стандартизация числовых полей
# print('Стандратизация данных')
# for col in digit_columns:
#     x[col] = (x[col] - x[col].mean()) / x[col].std()


print('Расчет корреляции для столбцов')

columns, correlations = [], []
for col in x.columns:
    columns.append(col)
    correlations.append(stats.pointbiserialr(x[col], Y)[0])

cd = pd.DataFrame({'column': columns, 'correlation': correlations})

print('Прямая корреляция')
print(cd.sort_values('correlation', ascending=False).head(30))
print('Обратная корреляция')
print(cd.sort_values('correlation', ascending=True).head(30))

# print('Графики зависимостей наиболее коррелирующих')
# for index, row in cd.sort_values('correlation', ascending=False).head(5).iterrows():
#     plt.title(row['column'])
#     sns.scatterplot(x = x[row['column']], y = Y)
#     plt.show()

# print('Графики зависимостей обратно коррелирующих')
# for index, row in cd.sort_values('correlation', ascending=True).head(5).iterrows():
#     plt.title(row['column'])
#     sns.scatterplot(x = x[row['column']], y = Y)
#     plt.show()

print('Обучение модели классификатора случайного леса')

rf = RandomForestClassifier(n_estimators=100, random_state=229)
rf.fit(x, Y)

importances = pd.Series(rf.feature_importances_, index=x.columns)
# X_selected = x[importances[importances].index]
print(importances.sort_values(ascending=False).head(30))
print(importances.sort_values(ascending=True).head(30))

# print('Графики зависимостей наиболее коррелирующих')
# for index, row in zip(importances.sort_values(ascending=False).head(7).index, importances.sort_values(ascending=False).head(7)):
# #     plt.title(index)
# #     sns.scatterplot(x = x[index], y = Y)
# #     plt.show()

# с помощью параметра hue разделим соответствующие классы целевой переменной
sns.scatterplot(x = x['variable_7'], y = x['variable_9'], hue = Y)
# добавим легенду, зададим ее расположение и размер
plt.legend(loc = 'upper left', prop = {'size': 15})
# выведем результат
plt.show()


# с помощью параметра hue разделим соответствующие классы целевой переменной
sns.scatterplot(x = x['variable_4'], y = x['variable_12'], hue = Y)
# добавим легенду, зададим ее расположение и размер
plt.legend(loc = 'upper left', prop = {'size': 15})
# выведем результат
plt.show()

x = x[importances.head(10).index]



print(x.shape)
print('Расчет кореляций (после)')
correlations_data = x.corr()
print('Вывод тепловой карты')
sns.heatmap(correlations_data, 
            # annot=True, 
            cmap='coolwarm', 
            square=True, 
            # linewidths=0.5,
            fmt='.2f')
plt.show()
