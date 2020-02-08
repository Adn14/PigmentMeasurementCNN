# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:15:57 2018

@author: Adn
"""

import pandas as pd
import numpy as np

from sklearn import svm

import matplotlib.pyplot as plt
import seaborn as sns

recipes = pd.read_csv('muffin_cupcakes.csv')

atribut = recipes[['Sugar','Butter']].as_matrix()

type_label = np.where(recipes['Type'] == 'Muffin', 0, 1)

print(type_label)

model = svm.SVC(kernel = 'Linear')
model.fit(atribut, type_label)

new_recipes = [[10, 10]]

a = model.predict(new_recipes)
if(a == 0):
    print("Muffin")
else:
    print("Cupcake")    
    
sns.lmplot('Sugar', 'Butter', data = recipes, hue = 'Type', fit_reg = False)

w = model.coef_[0]
a = -w[0]/w[1]
x = np.linspace(5, 30)
y  = a * x - (model.intercept_[0]/w[1])

b = model.support_vectors_[0]
y1 = a * x + (b[1] - a * b[0])
plt.plot(x, y1, 'k--')


plt.plot(x, y, linewidth = 2, color = 'black')


plt.plot(10, 10, 'bo', markersize = '12')
plt.show()
