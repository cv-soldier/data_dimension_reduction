import mpl_toolkits.mplot3d

from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from numpy import *
from numpy import linalg as la
from PIL import Image
import os
import glob
from matplotlib import pyplot as plt

"""-------------------鸢尾花数据降维--------------------------"""


def pca(X, k):
    """
    鸢尾花特征分解方法实现数据降维
    :param X: 数据集
    :param k: 维度
    :return:
    """
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    norm_X = X - mean
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    eig_pairs.sort(reverse=True)
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    data = np.dot(norm_X, np.transpose(feature))
    return data


"""-----------------------人脸图像数据降维------------------------------"""


def loadImageSet(path):
    """
    加载图像，将图片转化为一维向量
    :param add:
    :return:
    """
    filenames = glob.glob(path)
    filenames.sort()
    img = [Image.open(fn).convert('L').resize((98, 116)) for fn in filenames]
    FaceMat = np.asarray([np.array(im).flatten() for im in img])
    return FaceMat

def file_path(directory_path):
    """
    读取图片数据
    :param directory_path:
    :return:
    """
    faces_date = []
    for filename in os.listdir(directory_path):
        path = directory_path + "//" + filename
        faces_date.append(loadImageSet(path))
    return faces_date

def ReconginitionVector(path, selecthr=0.8):
    """
    特征脸算法实现PCA
    :param path: 图片路径
    :param selecthr: 特征值
    :return:
    """
    # step1: load the face image data ,get the matrix consists of all image
    FaceMat = file_path(path)
    # step2: average the FaceMat 平均脸
    avgImg = mean(FaceMat, 0)
    # step3: calculate the difference of avgimg and all image data(FaceMat)
    # 差值图像的数据矩阵
    diffTrain = FaceMat - avgImg
    # 协方差矩阵
    covMat = np.asmatrix(diffTrain) * np.asmatrix(diffTrain.T)
    # 特征值和特征向量
    eigvals, eigVects = linalg.eig(covMat)  # la.linalg.eig(np.mat(covMat))
    # step4: calculate eigenvector of covariance matrix (because covariance matrix will cause memory error)
    # 排序
    eigSortIndex = argsort(-eigvals)
    for i in range(shape(FaceMat)[1]):
        if (eigvals[eigSortIndex[:i]] / eigvals.sum()).sum() >= selecthr:
            eigSortIndex = eigSortIndex[:i]
        break
    covVects = diffTrain.T * eigVects[:, eigSortIndex]  # covVects is the eigenvector of covariance matrix
    # avgImg 是均值图像，covVects是协方差矩阵的特征向量，diffTrain是偏差矩阵
    return avgImg, covVects


if __name__ == '__main__':

    """---------------------鸢尾花数据降维------------------------"""

    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    """------SVD实现PCA数据降维------"""

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
    X_reduced_1 = PCA(n_components=3).fit_transform(iris.data)
    ax.scatter(
        X_reduced_1[:, 0],
        X_reduced_1[:, 1],
        X_reduced_1[:, 2],
        c=y,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )

    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()

    """------特征分解法实现PCA数据降维------"""

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
    X_reduced = pca(iris.data, 3)
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=y,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )

    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()

    """ -----------------------人脸图像数据降维----------------------------"""

    # 导入人脸
    path = "E:\\data_dimension_reduction\\face_data"
    Face = file_path(path)
    # 获取平均脸和特征向量
    avgImg, covVects = ReconginitionVector(path, selecthr=0.8)
    plt.imshow(np.mean(Face, 0).reshape(116, 98), cmap="gray")
    # 特征脸
    plt.show()

    # 待识别的15张图
    fig, axes = plt.subplots(3, 5, subplot_kw={"xticks": [], "yticks": []})  # 15张图，数量不一样修改，下同
    for i, ax in enumerate(axes.flat):
        ax.imshow(Face[i].reshape(116, 98), cmap="gray")
    plt.show()

    # 待识别15张图的特征脸
    fig, axes = plt.subplots(3, 5, subplot_kw={"xticks": [], "yticks": []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(covVects[:, i].reshape(116, 98), cmap="gray")
    plt.show()
