# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
pd.set_option('display.max_columns',None)


def deal_data(df):
    print(df.isnull().sum())
    df['CO'] = df['CO'].astype(int)
    print(df.info())
    plt.pie(df.iloc[:, 1:].sum(), labels=['AQI', 'PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3_8h'],textprops={'fontsize':20})
    plt.title('各个特征的占比', family='KaiTi', fontsize=30, loc='center')
    plt.show()
    print(df[(df['AQI'] == 0) & (df['PM2.5'] == 0) & (df['PM10'] == 0)])
    df = df.drop(df[(df['AQI'] == 0) & (df['PM2.5'] == 0) & (df['PM10'] == 0)].index)
    tran = StandardScaler()
    x=tran.fit_transform(df.iloc[0:, 1:])
    return x

def features_work(x):
    pca = PCA(n_components='mle')#mle 最大似然估计
    x_pca = pca.fit_transform(x)
    good_com=len(x_pca[0])
    #保留前几个主成分
    pca = PCA().fit(x)
    act=pca.explained_variance_ratio_
    act_sum=np.cumsum(act)
    plt.plot([1,2,3,4,5,6,7],act_sum)
    plt.xticks([1,2,3,4,5,6,7],fontsize = 20)
    plt.yticks(fontsize = 15)
    plt.title('特征数目占比',fontsize = 20,family = 'KaiTi')
    plt.xlabel('特征选取数量',family = 'KaiTi',fontsize = 25)
    plt.ylabel('特征在原始信息的占比',family = 'KaiTi',fontsize = 25)
    plt.show()
    return good_com



def create_model(x,n):
    sil,temp,preds,type=[],[],[],0
    k = [3,4,5,6,7]
    pca = PCA(n_components=n).fit_transform(x)
    for j in range(len(k)):
            estamtor=KMeans(n_clusters=k[j])
            estamtor.fit(pca)
            pred=estamtor.predict(pca)
            preds.append(pred)
            temp.append(silhouette_score(pca, pred))
            sil.append(k[j])
            print('聚类为'+str(j+3)+'的轮廓系数:',silhouette_score(pca,pred))
    fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1)
    fig.set_size_inches(10, 10)
    for i,ax in enumerate((ax1,ax2,ax3,ax4,ax5)):
        ax.scatter(pca[0:,0],pca[0:,1],c=preds[i])
        ax.set_title(str(k[i])+'种聚类情况' , family='KaiTi', loc='center')

    plt.show()
    plt.plot(k,temp)
    plt.title('kmeans簇类中心数目参数比较',family='KaiTi',fontsize = 25)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize =15)
    plt.xlabel('簇类数目',family='KaiTi',fontsize = 25)
    plt.ylabel('轮廓系数',family='KaiTi',fontsize = 25)
    plt.show()





if __name__ == '__main__':
    df = pd.read_excel('data.xlsx')
    df=deal_data(df)
    n=features_work(df)
    create_model(df,n)
