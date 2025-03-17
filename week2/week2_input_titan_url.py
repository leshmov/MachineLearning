import pandas as pd
from sklearn.model_selection import train_test_split #학습용 테스트용 나눌때 사용용
from sklearn.tree import DecisionTreeClassifier #DTree
from sklearn.ensemble import RandomForestClassifier #Randomforest
from sklearn.svm import SVC #SVC
from sklearn.linear_model import LogisticRegression #LR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #평가
import matplotlib.pyplot as plt

# url= "https://raw.github.com/MyungKyuYi/AI-class/blob/main/titanic.csv"
url = "https://raw.githubusercontent.com/MyungKyuYi/AI-class/main/titanic.csv"
# ㄴ 마찬가지로 raw를 붙여서 html이 아니라 csv 파일을 반환하게 함.
# ㄴ raw.githubusercontent.com 는 실제 데이터를 받을수있음.

url_df = pd.read_csv(url)

print(url_df.head())  # 첫 행 출력
print(url_df.columns)  # 컬럼 이름 출력
