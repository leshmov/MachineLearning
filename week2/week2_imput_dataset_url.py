import pandas as pd
# from sklearn.model_selection import train_test_split #학습용 테스트용 나눌때 사용용
# from sklearn.tree import DecisionTreeClassifier #DTree
# from sklearn.ensemble import RandomForestClassifier #Randomforest
# from sklearn.svm import SVC #SVC
# from sklearn.linear_model import LogisticRegression #LR
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #평가
# import matplotlib.pyplot as plt

url= "https://raw.githubusercontent.com/MyungKyuYi/AI-class/main/combined_dataset-1.xlsx" 
# ㄴ raw 를 빼고 가져올경우 html 을 반환하므로 raw를 붙여서 엑셀파일을 반환하도록 만들기 
url_df = pd.read_excel(url, engine = "openpyxl")
# ㄴ openpyxl 엑셀파일을 읽고 처리할때 필요함

print(url_df.head())  # 첫 행 출력
print(url_df.columns)  # 컬럼 이름 출력