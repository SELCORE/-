해당 파일을 사용하기 전 설치해야할게 있다.

먼저 virtualenv를 설치한다

pip install virtualenv

프로젝트 디렉토리로 이동하여 가상 환경을 만든다

cd /path/to/your/project

virtualenv venv

가상환경 활성화 - 이건 운영체제마다 다르니 주의하자

windows - venv\Scripts\activate

masOS/Linux - source venv/bin/activate

이제 라이브러리를 설치해야한다.

내가 사용한 라이브러리는 다음과 같다.

pandas

numpy

scikit-learn

matplotlib

seaborn

프로젝트 디렉토리는 다음과 같이 구성해야한다.

my_project/

│

├── venv/                 # 가상 환경 디렉토리

├── data.csv              # 데이터 파일

├── complex_lung_cancer_prediction.py  # Python 스크립트

└── requirements.txt      # 필요한 라이브러리 목록


