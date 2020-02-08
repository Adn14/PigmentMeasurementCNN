import csv
from csv import reader
import numpy as np
import matplotlib.pyplot as plt
# Import pandas library 
import pandas as pd 
import hashlib
def getProject(data):

    project = list()

    for i in range(len(data)):
        project.append(data[i][0])

    project = list(set(project))

    return project

def load_csv(filename):
    
    data= list()
    with open(filename, 'r', encoding='utf-8-sig') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data.append(row)
    return data

def load_records(data, project):
    rec = list()
    for i in data:
        if i[0] == project:
            rec.append(i[1:5])

    return rec


files = ['rgb', 'rgb_hsv', 'rgb_lab', 'rgb_ycbcr', 'hsv', 'hsv_lab', 'hsv_ycbcr', 'lab', 'lab_ycbcr', 'ycbcr']
path = "static/project_record/"

#load all each color scoring
allRgb = getProject(np.array(load_csv(path + files[0] + '.csv')))
allRgbHsv = getProject(np.array(load_csv(path + files[1] + '.csv')))
allRgbLab = getProject(np.array(load_csv(path + files[2] + '.csv')))
allRgbYcb = getProject(np.array(load_csv(path + files[3] + '.csv')))
allHsv = getProject(np.array(load_csv(path + files[4] + '.csv')))
allHsvLab = getProject(np.array(load_csv(path + files[5] + '.csv')))
allHsvYcb = getProject(np.array(load_csv(path + files[6] + '.csv')))
allLab = getProject(np.array(load_csv(path + files[7] + '.csv')))
allLabYcb = getProject(np.array(load_csv(path + files[8] + '.csv')))
allYcb = getProject(np.array(load_csv(path + files[9] + '.csv')))

allRgbIp = getProject(np.array(load_csv(path + files[0] + '_inpaint.csv')))
allRgbHsvIp = getProject(np.array(load_csv(path + files[1] + '_inpaint.csv')))
allRgbLabIp = getProject(np.array(load_csv(path + files[2] + '_inpaint.csv')))
allRgbYcbIp = getProject(np.array(load_csv(path + files[3] + '_inpaint.csv')))
allHsvIp = getProject(np.array(load_csv(path + files[4] + '_inpaint.csv')))
allHsvLabIp = getProject(np.array(load_csv(path + files[5] + '_inpaint.csv')))
allHsvYcbIp = getProject(np.array(load_csv(path + files[6] + '_inpaint.csv')))
allLabIp = getProject(np.array(load_csv(path + files[7] + '_inpaint.csv')))
allLabYcbIp = getProject(np.array(load_csv(path + files[8] + '_inpaint.csv')))
allYcbIp = getProject(np.array(load_csv(path + files[9] + '_inpaint.csv')))



path = 'static/project_record/model_select.csv'

selected = np.array(load_csv(path))



import os

p = 'static/model/sand-01/no_inpaint/hsvlab/'



models = list()

for file in os.listdir(p):
    if file.endswith(".h5"):
    	file = file.replace('.h5', '')
    	models.append(file)


mod = 'model_hsv_lab_2019-05-30_20-54-22.898348'

split = "_"

import re

after = '\\b'+'hsv_lab'+ '_' + '\\b'

result = re.split(after,mod)[-1]

print(mod[mod.index(split) + len(split):])



import substring

s = substring.substringByChar(mod, startChar="b", endChar="")

pos = [m.start() for m in re.finditer(r"_",mod)]
posStripe = [m.start() for m in re.finditer(r"-",mod)]

substr = mod[pos[2] + 1:]
substr2 = mod[:pos[2]]
clock = substr = mod[pos[3] + 1:]
clock = clock.replace('-', ':')

posTitik = [m.start() for m in re.finditer(r"'.'",mod)]
print(mod)
print(posTitik)
print('after clock part')



path = 'model_select.csv'


selected = np.array(load_csv(path))



c = 0

for i in selected:
	c += 1

def load_model(model): #load model
    model = model.replace('static/model/', '')
    print('Loading ', model)
    json_file = open('static/model/'+model+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("static/model/"+model+".h5")
    print('Finished loading ', model)

    return loaded_model

#print(selected)

def read_model_path(paths):
	path = list()
	for i in paths:
		if i[1] == 'yes':
			inpaint = 'inpaint/'
		else:
			inpaint = 'no_inpaint/'

		warna = i[0].replace('_', '')
		file = i[3] + '_' + i[4].replace(':', '-')
		string = 'static/model/' + i[2] + '/' + inpaint  + warna + '/' +  file
		path.append(string)

	return path


str = "GeeksforGeeks"
  
# encoding GeeksforGeeks using encode() 
# then sending to md5() 
result = hashlib.md5(str.encode()) 
  
# printing the equivalent hexadecimal value. 
print("The hexadecimal equivalent of hash is : ", end ="") 
print(result.hexdigest())

filename = "static/project_record/hsv_coba.csv"
data = load_csv(filename)

print(len(data))



'''file = "static/project_record/projects.csv"
with open(file, mode='w', newline='') as score:
    score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(data)):
        score_writer.writerow([i+1, data[i][0], 1]) 
    '''

nilai = list()

for i in data:
    nilai.append(i)

ts = data[0][5]
ts = ts.replace('_', '')
ts = ts.replace('-', '')
ts = ts.replace(':', '')


nData = []
nData.append(data[0][0])
nData.append(data[0][1])
nData.append(data[0][2])
nData.append(data[0][3])
nData.append(data[0][4])
nData.append(data[0][5])
print(data[0][1])
print(nilai[0])
print(nData)
print(ts)
'''

filename = "static/project_record/hsv_coba.csv"
data = load_csv(filename)

files = ['rgb', 'rgb_hsv', 'rgb_lab', 'rgb_ycbcr', 'hsv', 'hsv_lab', 'hsv_ycbcr', 'lab', 'lab_ycbcr', 'ycbcr']
warna = ['RGB', 'RGB_HSV', 'RGB_LAB', 'RGB_YCbCr', 'HSV', 'HSV_LAB', 'HSV_YCbCr', 'LAB', 'LAB_YCbCr', 'YCbCr']
path = "static/project_record/"

allData = []
projPath = "static/project_record/projects.csv"
projData = load_csv(projPath)

warnaPath = "static/project_record/warna.csv"
warnaData = load_csv(warnaPath)
count = 0

for i in range(len(files)):
    filePath = path + files[i] + '.csv'    
    data = load_csv(filePath)
    count += 1
    for j in data:
        row = []
        color = 'no_color'
        proj = 'no_id'
        for l in warnaData:
            if warna[i] == l[1]:
                color = l[0]
        for k in projData:
            if k[1] == j[0]:
                proj = k[0]
        row.append(1)
        row.append(color) 
        row.append(proj) 
        row.append('no')       
        row.append(j[1])
        row.append(j[2])
        row.append(j[3])
        row.append(j[4])
        row.append(j[5])
        allData.append(row)

for i in range(len(files)):
    filePath = path + files[i] + '_inpaint.csv'    
    data = load_csv(filePath)
    count += 1
    for j in data:
        row = []        
        color = 'no_color'
        proj = 'no_id'
        for l in warnaData:
            if warna[i] == l[1]:
                color = l[0]
        for k in projData:
            if k[1] == j[0]:
                proj = k[0]
        row.append(1)
        row.append(color) 
        row.append(proj)
        row.append('yes')     
        row.append(j[1])
        row.append(j[2])
        row.append(j[3])
        row.append(j[4])
        row.append(j[5])
        allData.append(row)

print(allData[0])
        
print(len(allData))
print(count)

file = "static/project_record/all_record.csv"

f = float('20190604152722.526590')
print(f)


with open(file, mode='w', newline='') as score:
    score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(allData)):
        ts = allData[i][8]
        ts = ts.replace('-', '')
        ts = ts.replace('_', '')
        ts = ts.replace(':', '')
        ts = ts.split(".")[0]
        score_writer.writerow([allData[i][0], allData[i][1], allData[i][2],
            allData[i][3], allData[i][4], allData[i][5], allData[i][6],
            allData[i][7], allData[i][8], ts]) 
            '''

'''from operator import itemgetter


data = load_csv(file)


with open(file, mode='rt', newline='') as f, open('static/project_record/sorted.csv', 'w', newline='') as final:
    writer = csv.writer(final, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    _ = next(reader)
    
    sorted1 = sorted(reader, key=lambda row: float(row[9]))
    
    
    for row in sorted1:
        writer.writerow(row)

scoPath = "static/project_record/all_record.csv"
akuPath = "static/project_record/nilai_akurasi.csv"
data = load_csv(scoPath)
count = 0

with open(akuPath, mode='w', newline='') as score:
    score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(data)):
        count += 1
        score_writer.writerow([count, data[i][0], data[i][1], data[i][2],
            data[i][3], data[i][4], data[i][5], data[i][6],
            data[i][7], data[i][8]]) 


data = load_csv(akuPath)

print(data[0])
print(len(data[0]))

modelPath = "static/project_record/model.csv"
modelData = load_csv(modelPath)

print(modelData[0])
'''
filename = "static/project_record/rgb.csv"
data = load_csv(filename)

print(data)


creator = data[0][6]
time = data[0][5]
modelName = 'model_inpaint_'

print(len(data[0]))

print(creator)
print(time.replace(':', '-'))