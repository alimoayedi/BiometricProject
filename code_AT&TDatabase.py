import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import random



img_array = []
img_LBP = []
LBP_features = []
LBP_features_labels = []

# LBP parameters
num_points = 9
radius = 1

# labels
labels = ['m', 'm', 'm', 'm', 'm', 'm', 'm', 'f', 'm', 'f',
          'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm',
          'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm',
          'm', 'f', 'm', 'm', 'f', 'm', 'm', 'm', 'm', 'm']

# read images
for person in range(1, 41):
    label = labels[person-1]
    for image in range(1, 11):
        url = "D:/Biometric Project/AT&T Database/s" + str(person) + "/" + str(image) + ".pgm"
        with open(url, 'rb') as pgmf:
            im = plt.imread(pgmf)
            # add left column of zero
            col_zero = np.zeros((im.shape[0], 1))
            im = np.concatenate((col_zero, im), 1)
            # add row zero at index 0
            row_zero = np.zeros((1, im.shape[1]))
            im = np.concatenate((row_zero, im), 0)
            # add right column of zero
            col_zero = np.zeros((im.shape[0], 1))
            im = np.concatenate((im, col_zero), 1)
            # add last row of zero
            row_zero = np.zeros((1, im.shape[1]))
            im = np.concatenate((im, row_zero), 0)

            # save all the raw images
            img_array.append([im, label])

            # calculate LBP
            lbp = feature.local_binary_pattern(im, num_points, radius, method="uniform")
            # save LBP matrix
            img_LBP.append(lbp)
            hist = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 2), range=(0, num_points + 2))[0].astype("float")
            # normalize the histogram
            hist /= (hist.sum() + 0.1)
            LBP_features.append(list(hist))
            LBP_features_labels.append(label)
            
    
train_used_index = []
test_used_index = []

train = []
test = []
train_labels = []
test_labels = []


for _ in range(0, 280):
    rand_index = random.randint(0, 399)
    while (rand_index in train_used_index):
        rand_index = random.randint(0, 399)
    train_used_index.append(rand_index)   
    train.append(LBP_features[rand_index])
    train_labels.append(LBP_features_labels[rand_index])
        
male = 0
female = 0
for label in train_labels:
    if (label == 'm'):
        male += 1 
    else:
        female += 1
        
for _ in range(0,120):
    rand_index = random.randint(0, 399)
    while (rand_index in train_used_index or rand_index in test_used_index):
        rand_index = random.randint(0, 399)
    test_used_index.append(rand_index)   
    test.append(LBP_features[rand_index])
    test_labels.append(LBP_features_labels[rand_index])
    
 
male = 0
female = 0
for label in test_labels:
    if (label == 'm'):
        male += 1 
    else:
        female += 1   
 

model = LinearSVC(C=100.0, random_state=42, max_iter=1000000)
model.fit(train, train_labels)
prediction = model.predict(np.array(test))

clf = MLPClassifier(activation="logistic", solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), max_iter=100000, random_state=1)
clf.fit(train, train_labels)
predict_labels = clf.predict(test)

conf_matrix = metrics.confusion_matrix(test_labels, predict_labels)

TP = conf_matrix[1][1]
TN = conf_matrix[0][0]
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

# calculate accuracy
conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
print('accuracy: ', conf_accuracy)

# calculate the sensitivity
conf_sensitivity = (TP / float(TP + FN))
print('sensitivity: ', conf_sensitivity)




