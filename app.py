import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import chardet
import os
import matplotlib.font_manager as fm

# 인코딩을 자동으로 감지하는 함수
def detect_encoding(file):
    result = chardet.detect(file.read())  # 파일을 읽고 인코딩 감지
    encoding = result['encoding']
    return encoding

# 사용자 정의 폰트 등록 함수
@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd() + '/custom_fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)

def main():
    fontRegistered()
    plt.rc('font', family='NanumGothic')

    st.title('K-Means Clustering App!!')

    # 1. CSV 파일 업로드
    file = st.file_uploader('CSV 파일 업로드', type=['csv'])

    if file is not None:
        # 파일 인코딩 확인
        encoding = detect_encoding(file)
        st.info(f'파일 인코딩: {encoding}')

        # 파일 포인터를 처음으로 돌려놓기 위해서 다시 `seek(0)`을 호출
        file.seek(0)

        # 2. 데이터 불러오기 (인코딩을 감지한 후 읽어오기)
        try:
            df = pd.read_csv(file, encoding=encoding, sep=' ', engine='python')
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"파일 읽기 오류: {str(e)}")
            return

        # NaN 값 처리
        st.info('Nan 이 있으면 행을 삭제합니다.')
        st.dataframe(df.isna().sum())
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 3. 유저가 컬럼을 선택할 수 있게 함
        st.info('K-Means 클러스터링에 사용할 컬럼을 선택해주세요.')
        selected_columns = st.multiselect('컬럼 선택', df.columns)

        if len(selected_columns) == 0:
            return

        df_new = pd.DataFrame()
        # 4. 각 컬럼이, 문자열인지 숫자인지 확인
        for column in selected_columns:
            if is_integer_dtype(df[column]):
                df_new[column] = df[column]
            elif is_float_dtype(df[column]):
                df_new[column] = df[column]
            elif is_object_dtype(df[column]):
                if df[column].nunique() <= 2:
                    # 레이블 인코딩
                    encoder = LabelEncoder()
                    df_new[column] = encoder.fit_transform(df[column])
                else:
                    # 원핫 인코딩
                    ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
                    column_names = sorted(df[column].unique())
                    df_new[column_names] = ct.fit_transform(df[column].to_frame())

        # K-Means 클러스터링을 위한 데이터 프레임 출력
        st.info('K-Means를 수행하기 위한 데이터 프레임입니다.')
        st.dataframe(df_new)

        # WCSS를 사용하여 최적의 k값 찾기
        st.subheader('최적의 k값을 찾기 위해 WCSS를 구합니다.')
        st.text(f'데이터의 갯수는 {df_new.shape[0]}개 입니다.')
        if df_new.shape[0] < 10:
            max_k = st.slider('K값 선택(최대 그룹갯수)', min_value=2, max_value=df_new.shape[0])
        else:
            max_k = st.slider('K값 선택(최대 그룹갯수)', min_value=2, max_value=10)

        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=4)
            kmeans.fit(df_new)
            wcss.append(kmeans.inertia_)

        fig1 = plt.figure()
        plt.plot(range(1, max_k + 1), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('클러스터 갯수')
        plt.ylabel('WCSS 값')
        st.pyplot(fig1)

        # 클러스터링(그룹) 갯수를 입력받아 최종 클러스터링 수행
        st.text('원하는 클러스터링(그룹) 갯수를 입력하세요')
        k = st.number_input('숫자 입력', min_value=2, max_value=max_k)

        kmeans = KMeans(n_clusters=k, random_state=4)
        df['Group'] = kmeans.fit_predict(df_new)

        st.info('그룹 정보가 저장되었습니다.')
        st.dataframe(df)

if __name__ == '__main__':
    main()
