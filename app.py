import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

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

    st.title('K-Means Clustering App')

    # 1. CSV 파일 업로드
    file = st.file_uploader('CSV 파일 업로드', type=['csv'])

    if file is not None:
        # 2. 데이터 불러오기(pandas)
        df = pd.read_csv(file)
        st.subheader("데이터 미리보기")
        st.dataframe(df.head())

        st.info('Nan 값이 있으면 행을 삭제합니다.')
        st.dataframe(df.isna().sum())  # 비어있는 데이터 확인
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        st.info('K-Means 클러스터링에 사용할 컬럼을 선택해주세요.')
        selected_columns = st.multiselect('컬럼 선택', df.columns)

        if len(selected_columns) == 0:
            st.warning("하나 이상의 컬럼을 선택해주세요.")
            return

        df_new = pd.DataFrame()

        # 3. 각 컬럼에 대해 데이터 타입을 확인하고 처리
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
            else:
                st.text(f'{column} 컬럼은 K-Means에 사용 불가하므로 제외하겠습니다.')

        st.info('K-Means 클러스터링을 위한 데이터 프레임')
        st.dataframe(df_new)

        st.subheader('최적의 k 값을 찾기 위해 WCSS를 구합니다.')

        # 최대 k 값은 데이터 개수에 따라 제한
        st.text(f'데이터의 갯수는 {df_new.shape[0]}개 입니다.')
        max_k = st.slider('K값 선택(최대 그룹 갯수)', min_value=2, max_value=min(10, df_new.shape[0]))

        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=4)
            kmeans.fit(df_new)
            wcss.append(kmeans.inertia_)

        # 차트 출력
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='-', color='b')
        plt.title('The Elbow Method', fontsize=14)
        plt.xlabel('클러스터 갯수', fontsize=12)
        plt.ylabel('WCSS 값', fontsize=12)
        st.pyplot(fig1)

        st.text('원하는 클러스터링(그룹) 갯수를 입력하세요.')
        k = st.number_input('숫자 입력', min_value=2, max_value=max_k)

        kmeans = KMeans(n_clusters=k, random_state=4)
        df['Group'] = kmeans.fit_predict(df_new)

        st.success('그룹 정보가 저장되었습니다.')
        st.dataframe(df)

if __name__ == '__main__':
    main()