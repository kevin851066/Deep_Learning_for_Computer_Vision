import cv2
import time
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Choose the patches randomly in the dataset
directory_name = ['tractor', 'reef', 'fountain', 'banana']
random_pics_label = np.random.randint(1, 501, size=4)
for i in range(4):
    file_name = directory_name[i] + '_' + str(random_pics_label[i]).zfill(3) + '.JPEG'
    img = cv2.imread(directory_name[i] + '/' + file_name)
    h, w, c = img.shape
    for j in range(4):
        for k in range(4):
            tile = img[j * int(h/4): (j+1) * int(h/4), k * int(w/4): (k+1) * int(w/4), :]
            cv2.imwrite(directory_name[i] + '/' + directory_name[i] + '_patches/{}_patch_{}.JPEG'.format(random_pics_label[i], 4*j + k), tile)

def zeroMean(dataMat):   # copy from https://blog.csdn.net/u012162613/article/details/42177327   
    meanVal = np.mean(dataMat, axis = 0)    
    newData = dataMat - meanVal
    return newData, meanVal

def pca(dataMat,n):   # copy from https://blog.csdn.net/u012162613/article/details/42177327
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar = 0) 
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) 
    eigValIndice = np.argsort(eigVals)                
    n_eigValIndice = eigValIndice[-1 : - (n + 1) : -1] 
    n_eigVect = eigVects[:, n_eigValIndice]
    lowDDataMat = newData * n_eigVect
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal
    return lowDDataMat, reconMat, n_eigVect, meanVal

def SliceAndFlatten(slice_list, img): # the number of tiles is 16
    h, w, _ = img.shape
    for i in range(4):
        for j in range(4):
            slice_list.append(img[i * int(h/4): (i+1) * int(h/4), j * int(w/4): (j+1) * int(w/4), :].flatten())

x_train, x_train_patches, x_train_label, x_test, x_test_patches, x_test_label = [], [], [], [], [], []

directory_name = ['tractor', 'reef', 'fountain', 'banana']
label = 0
for dir_name in directory_name:
    for i in range(500):
        file_name = dir_name + '_' + str(i+1).zfill(3) + '.JPEG'
        if i < 375:
            img = cv2.imread(dir_name + '/' + file_name)
            x_train.append(img.flatten())
            SliceAndFlatten(x_train_patches, img)
            x_train_label.append(label)
        else:
            img = cv2.imread(dir_name + '/' + file_name)
            x_test.append(img.flatten())
            SliceAndFlatten(x_test_patches, img)
            x_test_label.append(label)
    label = label + 1

x_train, x_train_patches, x_train_label, x_test, x_test_patches, x_test_label = np.array(x_train), np.array(x_train_patches), np.array(x_train_label), np.array(x_test), np.array(x_test_patches), np.array(x_test_label)

# K-Means clustering
start_time = time.time()
kmeans = KMeans(n_clusters=15, max_iter=5000)
label = kmeans.fit_predict(x_train_patches)
end_time = time.time()
print('time: ', end_time - start_time)
# dimension reduction
lowDim_x_train_patches, _, eigVec, mean = pca(x_train_patches, 3) # eigVec: (768, 3) 

lowDim_cluster_centers = eigVec.T * (kmeans.cluster_centers_ - mean).T 
lowDim_cluster_centers = lowDim_cluster_centers.T

lowDim_x_train_patches, lowDim_cluster_centers = np.real(lowDim_x_train_patches), np.real(lowDim_cluster_centers)

sample_clusters = random.sample(range(15), 6)
color_list = ['red', 'yellow', 'green', 'blue', 'orange', 'purple']

fig = plt.figure(figsize=(16,12))

ax = fig.add_subplot(111, projection='3d')

for i in range(6):
    coordinates = lowDim_x_train_patches[np.where(label==sample_clusters[i])]
    ax.scatter(lowDim_cluster_centers[sample_clusters[i], 0], 
                lowDim_cluster_centers[sample_clusters[i], 1], 
                lowDim_cluster_centers[sample_clusters[i], 2], s=50, marker='*', c='black')
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], s=1, c=color_list[i])

plt.show()


# compute the BoW of training images
def compute_dist(data, center):
    diff = data[:, np.newaxis, :] - center
    dist = np.sqrt(np.sum(np.square(diff), axis=2))
    return dist

def normalized_reciprocal_dist(dist):
    r = np.reciprocal(dist)
    s = np.sum(r, axis=1)
    nr_dist_mat = r / s[:, np.newaxis]
    return nr_dist_mat

kmeans = KMeans(n_clusters=15, max_iter=5000)
kmeans.fit(x_train_patches)
center = kmeans.cluster_centers_ # shape:(15, 16*16*3)

train_l2_dist_mat, test_l2_dist_mat = compute_dist(x_train_patches, center), compute_dist(x_test_patches, center) # shape:(24000, 15)
train_nr_dist_mat, test_nr_dist_mat = normalized_reciprocal_dist(train_l2_dist_mat), normalized_reciprocal_dist(test_l2_dist_mat)  # shape:(24000, 15)

BoW_train_features, BoW_test_features = [], []

for i in range(1500):
    train_bow = np.amax(train_nr_dist_mat[16*i : 16*(i+1) , :], axis=0)
    BoW_train_features.append(train_bow)

for j in range(500):
    test_bow = np.amax(test_nr_dist_mat[16*j : 16*(j+1) , :], axis=0)
    BoW_test_features.append(test_bow)

BoW_train_features, BoW_test_features = np.array(BoW_train_features), np.array(BoW_test_features)
# choose one image from each category and visualize its BoW using histogram plot.
x = np.arange(15)

for i in range(4):
    bow = BoW_features[6000*i]
    plt.bar(x, bow) 
    plt.savefig(directory_name[i] + '_bar')
    plt.show()

knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(BoW_train_features, x_train_label)

score = knn.score(BoW_test_features, x_test_label)

print(score)
     