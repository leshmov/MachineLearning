
import pandas as pd

# 붓꽃 데이터 CSV 파일 읽기
file_path = "C:/4-1/machineLearning/iris.csv"  # 본인이 iris.csv를 저장한 경로를 입력합니다.
df = pd.read_csv(file_path)

# 데이터프레임 확인
df.head()
df.columns()
