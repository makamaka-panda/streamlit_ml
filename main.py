import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st


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

def heatmap(dataframe, cols=[]]):
    st.header('ヒートマップ')
    st.subheader('特徴量を選択')
    numcols = tuple(dataframe.select_dtypes(exclude='object').columns)
    check = st.checkbox('全ての特徴量を選択', key = 0)
    if check:
        cols = list(numcols)
    select = st.multiselect('特徴量を選択してください', numcols, default=cols, key=0)
    if (len(select) > 1):
        sns.heatmap(dataframe[select].corr())
        return st.pyplot()
    else:
        return st.markdown('特徴量を二つ以上選択してください')

def pairplot(dataframe, cols=[]):
    st.header('散布図行列')
    st.subheader('数値の特徴量と色相にしたい特徴量を選択')
    numcols = dataframe.select_dtypes(exclude='object').columns
    catcols = ['色相にしたい説明変数を選択'] + list(dataframe.columns)
    check = st.checkbox('全ての変数を選択', key=1)
    if check:
        cols = list(numcols)
    select = st.multiselect('Numeric features:', tuple(numcols), default=cols, key=1)
    hue = st.selectbox('色相にする説明変数を選択', tuple(catcols), key=1)

    if (len(select) > 1):
        if hue == '色相にしたい説明変数を選択':
            sns.pairplot(dataframe[select])
            return st.pyplot()
        else:
            try:
                copy = select
                copy.append(hue)
                sns.pairplot(dataframe[copy], hue=hue)
                return st.pyplot()
            except:
                st.markdown("エラーが発生")

def violinplot(dataframe):
    st.header('バイオリンプロット')
    number_columns = tuple(dataframe.select_dtypes(exclude='object').columns)
    select1 = st.selectbox('数値の特徴量を選択', number_columns)
    all_columns = tuple(dataframe.columns)
    select2 = st.selectbox('特徴量を選択', all_columns)
    sns.violinplot(data=dataframe, x=select2, y=select1)
    return st.pyplot()


def valuecounts(dataframe):
    select = st.selectbox('Select one feature:', tuple(dataframe.columns))
    st.write(dataframe[select].value_counts())

def drop(dataframe, select):
    if len(select) != 0:
        return dataframe.drop(select, axis=1)
    else:
        return dataframe

def main():

    st.title('ランダムフォレストの検証')
    file = st.file_uploader('CSVファイルをアップロードしてください', type='csv')
    if file is not None:
        st.sidebar.header('設定')
        df_original = pd.DataFrame(readcsv(file))
        sidedrop = st.sidebar.multiselect('使用しない項目: ', tuple(df_original.columns))
        df = drop(df_original, sidedrop)
        model = st.sidebar.selectbox('回帰か分類かを選択', ('Regressor','Classifier'))
        target = st.sidebar.selectbox('目的変数を選択:', tuple(df.columns))

        st.header('CSVファイルの中身を表示')
        display_csv(df)
        st.header('要約統計量')
        st.dataframe(exploratory_func(df))
        st.header('各説明変数の値の数')
        valuecounts(df)

        show_figure = st.checkbox('図を表示するかどうか')
        if show_figure:
            heatmap(df)
            pairplot(df)
            violinplot(df)

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
            feat_importances.nlargest(10).plot(kind='barh', figsize=(8,8))
            st.pyplot()


if __name__ == '__main__':
    main()
