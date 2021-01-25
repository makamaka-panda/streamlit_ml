import streamlit as st
import pandas as pd
import seaborn as sns
#import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#import time

st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache
def readcsv(csv):
    df = pd.read_csv(csv)
    return df

def display_csv(dataframe):
    if len(dataframe) > 100:
        lenght = 100
    else:
        lenght = len(dataframe)
    slider = st.slider(f'表示させるレコード数（{len(dataframe)}レコード中最大100レコードまで表示可能）', min_value=5, max_value=lenght, value=5, step=5)
    st.dataframe(dataframe.head(slider))

@st.cache
def exploratory_func(dataframe):
    df = dataframe.describe().T
    df.insert(0, 'dtype', dataframe.dtypes.values)
    return df

def heatmap(dataframe, cols = []):
    st.header('Heatmap')
    st.subheader('Select numeric features:')
    numcols = tuple(dataframe.select_dtypes(exclude='object').columns)
    check = st.checkbox('Select all', key = 0)
    if check:
        cols = list(numcols)
    select = st.multiselect('Numeric features:', numcols, default=cols, key = 0)
    if (len(select) > 10):
        annot = False
    else:
        annot = True
    if (len(select) > 1):
        sns.heatmap(dataframe[select].corr(), annot=annot)
        return st.pyplot()

def pairplot(dataframe, cols = []):
    st.header('Pairplot')
    st.subheader('Select numeric features and 1 categorical feature at most')
    catcols = ['-']
    for i in list(dataframe.select_dtypes(include='object').columns):
        catcols.append(i)
    catcols = tuple(catcols)
    numcols = tuple(dataframe.select_dtypes(exclude='object').columns)
    check = st.checkbox('Select all', key = 1)
    if check:
        cols = list(numcols)
    select = st.multiselect('Numeric features:', numcols, default=cols, key = 1)
    hue = st.selectbox('Select hue', catcols, key = 0)

    if (len(select) > 1):
        if hue == '-':
            sns.pairplot(dataframe[select])
            return st.pyplot()
        else:
            try:
                copy = select
                copy.append(hue)
                sns.pairplot(dataframe[copy], hue = hue)
                return st.pyplot()
            except:
                st.markdown("An error occurred, please don't use hue for this dataframe")

def boxplot(dataframe):
    st.header('Select features for the Violinplot')
    numcol = tuple(dataframe.select_dtypes(exclude='object').columns)
    catcol = tuple(dataframe.select_dtypes(include='object').columns)
    select1 = st.selectbox('Selecione a numeric feature', numcol)
    select2 = st.selectbox('Selecione a categorical feature', catcol)
    sns.violinplot(data = dataframe, x = select2, y = select1)
    return st.pyplot()

def scatter(dataframe):
    st.header('Select features for the Scatterplot')
    numcol = tuple(dataframe.select_dtypes(exclude='object').columns)
    select1 = st.selectbox('Select numeric feature', numcol)
    select2 = st.selectbox('Select another numeric feature', numcol)
    plt.scatter(dataframe[select1], dataframe[select2])
    return st.pyplot()

def valuecounts(dataframe):
    select = st.selectbox('Select one feature:', tuple(dataframe.columns))
    st.write(dataframe[select].value_counts())

def drop(dataframe, select):
    if len(select) != 0:
        return dataframe.drop(select, 1)
    else:
        return dataframe

def main():

    st.title('ランダムフォレストの検証')
    # st.image('ia.jpg', use_column_width=True)
    file = st.file_uploader('CSVファイルをアップロードしてください', type='csv')
    if file is not None:

        st.sidebar.header('設定')
        st.sidebar.subheader('使用するグラフ')
        check_heatmap = st.sidebar.checkbox('Heatmap')
        check_pairplot = st.sidebar.checkbox('Pairplot')
        check_violinplots = st.sidebar.checkbox('Violinplots')
        check_scatterplot = st.sidebar.checkbox('Scatterplot')
        df0 = pd.DataFrame(readcsv(file))
        st.sidebar.subheader('使用しない項目')
        sidedrop = st.sidebar.multiselect('使用しない項目: ', tuple(df0.columns))
        df = drop(df0, sidedrop)
        st.sidebar.subheader('予測')
        model = st.sidebar.selectbox('回帰か分類かを選択', ('Regressor','Classifier'))
        target = st.sidebar.selectbox('目的変数を選択:', tuple(df.columns))

        st.header('CSVファイルの中身を表示')
        display_csv(df)
        st.header('要約統計量')
        st.dataframe(exploratory_func(df))
        st.header('各説明変数の値の数')
        valuecounts(df)

        if check_heatmap:
            heatmap(df)
        if check_pairplot:
            pairplot(df)
        if check_violinplots:
            boxplot(df)
        if check_scatterplot:
            scatter(df)

        st.header('ランダムフォレスト:')
        X = pd.get_dummies(df.drop(target,1))
        y = df[target]
        tt_slider = st.slider('% Size of test split:', min_value=1, max_value=99, value=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01*tt_slider, random_state=42)

        predict = st.button('予測')
        if predict:
            st.header('結果:')
            st.subheader('予測精度')
            if model == 'Regressor':
                rf = RandomForestRegressor()
            elif model == 'Classifier':
                rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            st.markdown(rf.score(X_test, y_test))
            #plt.figure(num=None, figsize=(6, 4), facecolor='w', edgecolor='k')
            feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
            st.subheader('影響力のある特徴量:')
            feat_importances.nlargest(10).plot(kind='barh', figsize = (8,8))
            st.pyplot()


if __name__ == '__main__':
    main()
