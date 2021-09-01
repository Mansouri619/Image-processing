import os
import cv2
import mahotas
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

fixed_size = tuple((500, 500))
bins = 8


# 7,13, 512

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


global_features = []
labels = []
train_labels = ['bluebell', 'crocus', 'daffodil', 'lilyvalley', 'snowdrop']

folder = '/Users/zahramansoori/Desktop/ImageProcessing2/images'
for name in train_labels:
    folder = folder + '/' + name
    for filename in os.listdir(folder):
        # print(filename)
        image = cv2.imread(os.path.join(folder, filename))
        image = cv2.resize(image, fixed_size)

        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)

        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        labels.append(name)
        global_features.append(global_feature)
    folder = '/Users/zahramansoori/Desktop/ImageProcessing2/images'


print("feature vector size {}".format(np.array(global_features).shape))


print("training Labels {}".format(np.array(labels).shape))


targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)


scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)


df_f = pd.DataFrame(rescaled_features)
df_l = pd.DataFrame(target)
print(df_f)
print(df_l)


def ML():
    x = df_f.iloc[:, :].values
    y = df_l.iloc[:, :].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    result = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(result)
    result1 = classification_report(y_test, y_pred)
    print("Classification Report:", )
    print(result1)
    result2 = accuracy_score(y_test, y_pred)
    print("Accuracy:", result2)


ML()
