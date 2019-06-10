import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.datasets import fetch_mldata
from sklearn.svm import LinearSVC


#loading the mnist original dataset
dataset = fetch_mldata("mnist-original" )

#storing the features of the image
features = np.array(dataset.data, 'int16')

#storing the labels of corresponding digits
labels = np.array(dataset.target, 'int')

#creation of list for storing histogram of gradient features of the image
list_hog_fd = []


for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

#creation of Linear support vector classifier object
clf = LinearSVC()

# preform the training using the fit member function of the clf object
clf.fit(hog_features, labels)

#saving the trained classifier in digits_cls.pkl file with compression parameter
joblib.dump(clf, "digits_cls.pkl", compress=3)


