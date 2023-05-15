# import numpy as np
# import pandas as pd
# import os
# import seaborn as sns
# import matplotlib.pyplot as plt
# import warnings
# import random as rnd
#
# from sklearn.ensemble import RandomForestClassifier
#
# DATA_IN_PATH = './titanic/'
# TRAIN_INPUT_DATA = 'train.csv'
# TEST_INPUT_DATA = 'test.csv'
#
# #train데이터를 데이터 프레임으로 만든 후 헤드만 출력
# df_train = pd.read_csv(DATA_IN_PATH + TRAIN_INPUT_DATA)
# print("train 데이터")
# print(df_train.head())
#
# #test데이터를 데이터 프레임으로 만든 후 헤드만 출력
# df_test = pd.read_csv(DATA_IN_PATH + TEST_INPUT_DATA)
# print("test 데이터")
# print(df_test.head())
#
# df = [df_train, df_test]
#
# #데이터 분석 - 개수, 평균 등을 알 수 있음
# print("데이터 분석")
# print(df_train.describe())
#
# #객실 등급에 따른 생존률 비교 -> 객실 등급이 높을 수록 생존률이 높음
# survived_groupby_pclass = df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# print("객실 등급에 따른 생존률 비교")
# print(survived_groupby_pclass)
# print()
#
# #성별에 따른 생존률 비교 -> 여성의 생존률이 남성 보다 높음
# survived_groupby_sex = df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# print("성별에 따른 생존률 비교")
# print(survived_groupby_sex)
# print()
#
# #함께 승선한 형제자매에 따른 생존률 비교 -> 형재재매가 적은 경우 생존률 높음
# survived_groupby_sibsp = df_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# print("형제자매에 따른 생존률 비교")
# print(survived_groupby_sibsp)
# print()
#
# #함께 승선한 부모, 자식수에 따른 생존률 비교 -> 부모 자식수가 적을수록 생존율이 높음
# survived_groupby_parch = df_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# print("부모, 자식수에 따른 생존률 비교")
# print(survived_groupby_parch)
# print()
#
# #연령에 따른 생존 여부
# survived_groupby_age = sns.FacetGrid(df_train, col='Survived')
# survived_groupby_age.map(plt.hist, 'Age', bins=20, color="skyblue")
# plt.show()
#
#
# #객실 등급과 생존 여부에 따른 연령 분포
# age_groupby_PclassSurvived = sns.FacetGrid(df_train, col='Survived', row='Pclass', hue='Pclass', height=2.2, aspect=1.6)
# age_groupby_PclassSurvived.map(plt.hist, 'Age', alpha=0.5, bins=20)
# age_groupby_PclassSurvived.add_legend()
# plt.show()
#
# #Embarked, Survived, Sex에 따른 Fare
# EmbarkedSurvivedSex_From_Fare = sns.FacetGrid(df_train, row='Embarked', col='Survived', height=2.2, aspect=1.6)
# EmbarkedSurvivedSex_From_Fare.map(sns.barplot, 'Sex', 'Fare', alpha=0.7, errorbar=None, order=['male','female'])
# EmbarkedSurvivedSex_From_Fare.add_legend()
# plt.show()
#
# #승선지와 객실등급에 따른 생존률 분포
# survived_groupby_EmbarkedPclass = sns.FacetGrid(df_train, row='Embarked', height=2.2, aspect=1.6)
# survived_groupby_EmbarkedPclass.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=[1, 2,3], palette='deep', hue_order=['male','female'])
# survived_groupby_EmbarkedPclass.add_legend()
# plt.show()
#
# #--------------------------------------------------------------------------------
# ##데이터 전처리##
#
# #Ticket, Cabin제거
# df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)
# df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)
# df = [df_train, df_test]
#
#
# #Mlle, Ms, Miss, Mrs는 여자로 / Master, Mr, Mme 는 남자로 이름 변경
# for dataset in df:
#     dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
#
# for dataset in df:
#     dataset['Title'] = dataset['Title'].replace(['Sir', 'Dona', 'Lady', 'Countess', 'Jonkheer', 'Capt', 'Col', 'Major', 'Don', 'Dr', 'Rev'], 'else')
#     dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
#     dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
#     dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
#
# print("호칭에 따른 생존율")
# print(df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
#
# #이름 앞에 붙는 호칭을 번호로 변경하기
# title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Else":5}
#
# for dataset in df:
#     dataset['Title'] = dataset['Title'].map(title_mapping)
#     dataset['Title'] = dataset['Title'].fillna(0)
#
#
# #안쓰는 변수 Name, PassengerId 제거
# df_train = df_train.drop(['Name', 'PassengerId'], axis=1)
# df_test = df_test.drop(['Name'],  axis=1)
# df = [df_train, df_test]
#
#
# #성별 변수를 숫자로 바꿔줌 (남자->0 / 여자->1)
# for dataset in df:
#     dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male':0}).astype(int)
#
#
# #나이가 적혀 있지 않는 사람의 나이 랜덤값으로 입력
# guess_age = np.zeros((2, 3))
# for dataset in df:
#     for i in range(0, 2):
#         for j in range(0, 3):
#             df_guess = dataset[(dataset['Sex']==i) & (dataset['Pclass']==j + 1)]['Age'].dropna()
#             age_guess = df_guess.median()
#             guess_age[i, j] = int(age_guess/0.5 + 0.5) * 0.5
#
#     for i in range(0, 2):
#         for j in range(0, 3):
#             dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_age[i, j]
#
#     dataset['Age'] = dataset['Age'].astype(int)
#
# # ageband를 바탕으로 age 변수를 범주형 변수로 변경
# df_train['AgeBand'] = pd.cut(df_train['Age'], 5)
# for dataset in df:
#     dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
#     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
#     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
#     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
#     dataset.loc[dataset['Age'] > 64, 'Age']
# df_train = df_train.drop(['AgeBand'], axis=1)
# df = [df_train, df_test]
#
# #SibSp 와 Parch를 가족과의 동반여부를 알 수 있는 새로운 변수로 통합
# for dataset in df:
#     dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
#
# #IsAlone변수가  1이면 가족을 동반하지 않음을 의미/ 0이면 가족을 동반했다는 것을 뜻함
# for dataset in df:
#     dataset['IsAlone'] = 0
#     dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
#
# df_train = df_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
# df_test = df_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
# df = [df_train, df_test]
#
# #Age변수와 Pclass를 곱한 Age*Class 변수 생성함
# for dataset in df:
#     dataset['Age*Class'] = dataset.Age * dataset.Pclass
#
# #Embarked변수의 결측치를 최빈값으로 대체함
# freq_port =  df_train.Embarked.dropna().mode()[0]
# for dataset in df:
#     dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
#
# #Embarked변수를 S = 0, C = 1, Q = 2의 정수형 변수로 바꿈
# for dataset in df:
#     dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
#
#
# #Fare변수의 결측치를 중앙값으로 대체
# df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)
# df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)
#
# #Fare변수를 범위에 따라 0, 1, 2, 3의 정수로 변경
# for dataset in df:
#     dataset.loc[dataset['Fare'] <= 8, 'Fare'] = 0
#     dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 15), 'Fare'] = 1
#     dataset.loc[(dataset['Fare'] > 15) & (dataset['Fare'] <= 31), 'Fare'] = 2
#     dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
#     dataset['Fare'] = dataset['Fare'].astype(int)
#
# df_train= df_train.drop(['FareBand'], axis=1)
# df = [df_train, df_test]
#
#
# #전처리된 최종 데이터 셋
# print("학습 데이터 셋")
# print(df_train.head())
#
# print("테스트 데이터 셋")
# print(df_test.head())
#
# #-------------------------------------------------------------------------------
# #데이터 준비
# X_train = df_train.drop("Survived", axis=1)
# Y_train = df_train['Survived']
# X_test = df_test.drop("PassengerId", axis=1).copy()
#
# ##Random Forest 모델 구축
# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
# Y_prediction = random_forest.predict(X_test)
# score = random_forest.score(X_train, Y_train)
# accuracy = round(score * 100, 2)
#
# print(f"정확도: {accuracy}")
#
#
#
#
#
#
#
#
#
#
#
#
