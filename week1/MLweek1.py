import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. 붓꽃 데이터 CSV 파일 읽기
file_path = "C:/4-1/ML/iris.csv"  # 본인이 iris.csv를 저장한 경로를 입력합니다.
df = pd.read_csv(file_path)

# 2. 데이터프레임 확인
print(df.head())  # 첫 5개 행 출력
print(df.columns)  # 컬럼 이름 출력

# 3. 특성과 타깃 변수 분리
# name 타겟 변수수
# 타깃 변수가 무엇인지 CSV 파일의 컬럼 이름을 확인한 후 수정해 주세요.
X = df.drop('Name', axis=1)  # 특성 데이터 (예: 'species' 컬럼 제외)
y = df['Name']  # 타깃 데이터 (예: 'species' 컬럼)

# 4. 데이터 분할 (학습용과 테스트용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 분류 모델 초기화
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

# 6. 모델 학습 및 평가
for name, model in models.items():
    model.fit(X_train, y_train)  # 모델 학습
    y_pred = model.predict(X_test)  # 예측
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("-" * 50)
