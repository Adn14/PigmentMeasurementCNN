import csv
from csv import reader
import numpy as np
import matplotlib.pyplot as plt
# Import pandas library 
import pandas as pd 

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

def rubah(kata):
	global word 
	word = kata

def sort_score(data):
	data = np.array(data)
	if len(data) > 0:
		arr = [np.array(np.sort(data[:,0])), np.array(np.sort(data[:,1])), np.array(np.sort(data[:,2])), np.array(np.sort(data[:,3]))]
		return arr
	return [[], [], [], []]

def get_avScore(data):

	if(len(data[0]) != 0):
		avLoss = float(0)
		avMae = float(0)
		avValLoss = float(0)
		avValMae = float(0)
		if(len(arr[0]) < 5):
			loop = len(data[0])
			
		else:
			loop = 5

		for i in range(loop):
			avLoss = avLoss + float(data[0][i])
			avMae = avMae + float(data[1][i])
			avValLoss = avValLoss + float(data[2][i])
			avValMae = avValMae + float(data[3][i])

		avLoss /= loop
		avMae /= loop
		avValLoss /= loop
		avValMae /= loop

		return [avLoss, avMae, avValLoss, avValMae]

	return [0, 0, 0, 0]
	


filename = "static/project_record/rgb.csv"
data = np.array(load_csv(filename))

rgb = list()

for i in data:
	print(i[0])
	rgb.append(i[1:4])


data = np.array(data)
data = data.flatten()

project = 'experiment_real_01'
files = ['rgb', 'rgb_hsv', 'rgb_lab', 'rgb_ycbcr', 'hsv', 'hsv_lab', 'hsv_ycbcr', 'lab', 'lab_ycbcr', 'ycbcr']
path = "static/project_record/"

allRgb = np.array(load_csv(path + files[0] + '.csv'))
allRgbHsv = np.array(load_csv(path + files[1] + '.csv'))
allRgbLab = np.array(load_csv(path + files[2] + '.csv'))
allRgbYcb = np.array(load_csv(path + files[3] + '.csv'))
allHsv = np.array(load_csv(path + files[4] + '.csv'))
allHsvLab = np.array(load_csv(path + files[5] + '.csv'))
allHsvYcb = np.array(load_csv(path + files[6] + '.csv'))
allLab = np.array(load_csv(path + files[7] + '.csv'))
allLabYcb = np.array(load_csv(path + files[8] + '.csv'))
allYcb = np.array(load_csv(path + files[9] + '.csv'))

allRgbIp = np.array(load_csv(path + files[0] + '_inpaint.csv'))
allRgbHsvIp = np.array(load_csv(path + files[1] + '_inpaint.csv'))
allRgbLabIp = np.array(load_csv(path + files[2] + '_inpaint.csv'))
allRgbYcbIp = np.array(load_csv(path + files[3] + '_inpaint.csv'))
allHsvIp = np.array(load_csv(path + files[4] + '_inpaint.csv'))
allHsvLabIp = np.array(load_csv(path + files[5] + '_inpaint.csv'))
allHsvYcbIp = np.array(load_csv(path + files[6] + '_inpaint.csv'))
allLabIp = np.array(load_csv(path + files[7] + '_inpaint.csv'))
allLabYcbIp = np.array(load_csv(path + files[8] + '_inpaint.csv'))
allYcbIp = np.array(load_csv(path + files[9] + '_inpaint.csv'))

rgb = load_records(allRgb, project)
rgbHsv = load_records(allRgbHsv, project)
rgbLab = load_records(allRgbLab, project)
rgbYcb = load_records(allRgbYcb, project)
hsv = load_records(allHsv, project)
hsvLab = load_records(allHsvLab, project)
hsvYcb = load_records(allHsvYcb, project)
lab = load_records(allLab, project)
labYcbcr = load_records(allLabYcb, project)
ycbcr = load_records(allYcb, project)

rgbIp = load_records(allRgbIp, project)
rgbHsvIp = load_records(allRgbHsvIp, project)
rgbLabIp = load_records(allRgbLabIp, project)
rgbYcbIp = load_records(allRgbYcbIp, project)
hsvIp = load_records(allHsvIp, project)
hsvLabIp = load_records(allHsvLabIp, project)
hsvYcbIp = load_records(allHsvYcbIp, project)
labIp = load_records(allLabIp, project)
labYcbcrIp = load_records(allLabYcbIp, project)
ycbcrIp = load_records(allYcbIp, project)


#rgb = np.array(rgb)
#lab = np.array(lab)
#hsv = np.array(hsv)
#print('length lab: ', len(lab))

#s = list()
#sa = np.array(np.sort(hsv[:,0])[::-1])

#rr = [np.array(np.sort(rgb[:,0])[::-1]), np.array(np.sort(rgb[:,1])[::-1]), np.array(np.sort(rgb[:,2])[::-1]), np.array(np.sort(rgb[:,3])[::-1])]
arr = sort_score(rgb)
#print(sort[:3])
#print(sa)
print(arr[0])
