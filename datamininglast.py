# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:17:18 2018

@author: Adn
"""
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB

# tinggi, berat, ukuran sepatu

x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 65, 39], 
     [177, 77, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

y = ['Pria', 'Pria', 'Wanita', 'Wanita', 'Pria', 'Pria', 'Wanita', 'Wanita', 'Wanita', 'Pria', 'Pria']

clf1 = tree.DecisionTreeClassifier()
clf2 = svm.SVC()
clf3 = neighbors.KNeighborsClassifier()
clf4 = GaussianNB()

clf1 = clf1.fit(x, y)
clf2 = clf2.fit(x, y)
clf3 = clf3.fit(x, y)
clf4 = clf4.fit(x, y)

newX = [[184, 84, 44], [198, 92, 48], [183, 83, 44], [166, 47, 36], [170, 60, 30], [172, 64, 39], [182, 80, 42],
        [180, 80, 43]]

newY = ['Pria', 'Pria', 'Pria',
        'Wanita', 'Wanita', 'Wanita',
        'Pria', 'Pria']

prediction1 = clf1.predict(newX)
prediction2 = clf2.predict(newX)
prediction3 = clf3.predict(newX)
prediction4 = clf4.predict(newX)

print("Tree : ", prediction1)
print("SVM : ",prediction2)
print("KNN : ",prediction3)
print("Bayes : ",prediction4)

from sklearn.metrics import accuracy_score
score1 = accuracy_score(newY, prediction1)
score2 = accuracy_score(newY, prediction2)
score3 = accuracy_score(newY, prediction3)
score4 = accuracy_score(newY, prediction4)

print("Decision Tree: ", score1*100, '%')
print("SVM: ", score2*100, '%')
print("KNN: ", score3*100, '%')
print("Bayes: ", score4*100, '%')