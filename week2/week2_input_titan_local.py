import pandas as pd
from sklearn.model_selection import train_test_split #학습용 테스트용 나눌때 사용용
from sklearn.tree import DecisionTreeClassifier #DTree
from sklearn.ensemble import RandomForestClassifier #Randomforest
from sklearn.svm import SVC #SVC
from sklearn.linear_model import LogisticRegression #LR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #평가
import matplotlib.pyplot as plt

local = "C:/4-1/ML/titanic.csv"  
local_df = pd.read_csv(local)

print(local_df.head())  # 첫 행 출력
print(local_df.columns)  # 컬럼 이름 출력
print(local_df.isnull().mean())  # 각 열의 결측치 비율