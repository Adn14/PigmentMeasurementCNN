from __future__ import print_function
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import  Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory, Response, redirect, url_for, jsonify, json
from csv import reader
import os
import cv2
import csv
import numpy as np

import pyrebase
import matplotlib.pyplot as plt
import pandas as pd 
import hashlib

__author__ = 'Adn'

PEOPLE_FOLDER = os.path.join('static')

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

ALLOWED_EXTENSIONS = set(['json'])

projectPath = "static/project_record/db/projects.csv"
userPath = "static/project_record/db/user.csv"
warnaPath = "static/project_record/db/warna.csv"
akurasiPath = "static/project_record/db/nilai_akurasi.csv"
modelSelPath = "static/project_record/db/model.csv"

config = {
    "apiKey": "AIzaSyAc5UAYhx9ecwRHxij3PMMD8R88l_6tuEU",
    "authDomain": "skripsi-4c7b7.firebaseapp.com",
    "databaseURL": "https://skripsi-4c7b7.firebaseio.com",
    "projectId": "skripsi-4c7b7",
    "storageBucket": "skripsi-4c7b7.appspot.com",
    "messagingSenderId": "137288406633",
    "appId": "1:137288406633:web:881af2919f4ae797"
}
ts = '' #main timestamp
global modelToSave
global modelJsonToSave
global dirToSave
global currentUserName 
currentUserName = ''             
global currentUserId
currentUserId = 0
global currentAuth 
currentAuth = 0
py_path = os.path.dirname(os.path.realpath(__file__))

userId = 0
userName = ''

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def create_dir(path):
    if not os.path.exists(path):
       os.makedirs(path)
       print(path + ' created')
    else:
        print('file already exists')

def load_csv(filename):
    
    data= list()
    with open(filename, 'r', encoding='utf-8-sig') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data.append(row)
    return data

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

def load_model(model): #load model
    print('model path :' , model)
    model = model.replace('static/model/', '')
    print('Loading ', model)
    json_file = open('static/model/'+model+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("static/model/"+model+".h5")
    print('Finished loading ', model)

    return loaded_model

def res_splitter(result): #split the estimation result into 3 variables
    maxAnto = float(89.7352)
    maxKaro = float(53.2421)
    maxKlor = float(1011.7251)
    res1 = float(result[0][0])
    res2 = float(result[0][1])
    res3 = float(result[0][2])

    res1 = float((res1*(maxAnto - 0) + 0))
    res2 = float((res2*(maxKaro - 0) + 0))
    res3 = float((res3*(maxKlor - 0.0063) + 0.0063))

    if(res1<0):
        res1 = 0
    if(res2<0):
        res2 = 0
    if(res3<0):
        res3 = 0

    return str(res1), str(res2), str(res3)

def image_appender(img): #make a 4 dimensional numpy array
    image = []
    image.append(img)
    image = np.array(image)
    image = image.astype(np.floating)
    image /= 255
    return image

def inpaint_image(image): #applying inpainting method to image
    channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    channel = cv2.equalizeHist(channel)
    ret, otsumask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsumask = cv2.bitwise_not(otsumask)
    im2, contours, hierarchy = cv2.findContours(otsumask, cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    c = max(contours, key = cv2.contourArea)
    
    for i in range(len(contours)):
        hull.append(cv2.convexHull(c, True))
        
    drawing = np.zeros((otsumask.shape[0], otsumask.shape[1], 3), np.uint8)
    
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv2.drawContours(drawing, contours, i, color, cv2.FILLED)
    
    mask_new = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    res = image.copy()
    res[mask_new == (0)] = (0, 0, 1)
    
    ret,thresh1 = cv2.threshold(channel,130,255,cv2.THRESH_BINARY)
    
    dst2 = cv2.inpaint(res, thresh1, 3, cv2.INPAINT_NS)
    
    rows, cols, channels = dst2.shape
    roi = image[0:rows, 0:cols]
    mask_inv = cv2.bitwise_not(mask_new)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_fg = cv2.bitwise_and(dst2,dst2,mask = mask_new)
    over = cv2.add(img1_bg,img2_fg)
    image[0:rows, 0:cols] = over
    return image    

def load_records(data, project):
    rec = list()
    for i in data:
        if i[0] == project:
            rec.append(i[1:7])

    return rec

def sort_score(data):
    data = np.array(data)
    if len(data) > 0:
        arr = [np.array(np.sort(data[:,0])), np.array(np.sort(data[:,1])), np.array(np.sort(data[:,2])), np.array(np.sort(data[:,3]))]
        c = np.array(sorted(data, key=lambda col: col[1]))[:,0:4]
        return c
    return [[], [], [], []]

def get_avScore(data):

    if(len(data[0]) != 0):
        avLoss = float(0)
        avMae = float(0)
        avValLoss = float(0)
        avValMae = float(0)
        if(data.shape[0] < 5):
            loop = int(data.shape[0])
            
        else:
            loop = 5

        for i in range(loop):
            avLoss = avLoss + float(data[i][0])
            avMae = avMae + float(data[i][1])
            avValLoss = avValLoss + float(data[i][2])
            avValMae = avValMae + float(data[i][3])

        avLoss /= loop
        avMae /= loop
        avValLoss /= loop
        avValMae /= loop

        return [avLoss, avMae, avValLoss, avValMae]

    return [0, 0, 0, 0]

def create_chart(data, project, metric, mode, title):
    fileName = project+'_'+metric+'_'+mode+'.png'
    df = pd.DataFrame(data, columns = ['Color', 'Training', 'Validation'])
    df.plot.bar(x = 'Color', y = ['Training', 'Validation'], rot = 40)
    plt.suptitle(project + title)
    plt.savefig('static/project_record/'+ fileName, bbox_inches="tight")
    return fileName

print('Commencing load model phase.')

modelCsvPath = 'static/project_record/model_select.csv'
modelCsvPaths = np.array(load_csv(modelCsvPath))
modelCsvPaths = read_model_path(modelCsvPaths)

#load inpaint model
modelIpRgb = load_model(modelCsvPaths[0])
modelIpRgbHsv = load_model(modelCsvPaths[1])
modelIpRgbLab = load_model(modelCsvPaths[2])
modelIpRgbYcbcr = load_model(modelCsvPaths[3])
modelIpHsv = load_model(modelCsvPaths[4])
modelIpHsvLab = load_model(modelCsvPaths[5])
modelIpHsvYcbcr = load_model(modelCsvPaths[6])
modelIpLab = load_model(modelCsvPaths[7])
modelIpLabYcbcr = load_model(modelCsvPaths[8])
modelIpYCbCr = load_model(modelCsvPaths[9])

#load non-inpaint model

modelRgb = load_model(modelCsvPaths[10])
modelRgbHsv = load_model(modelCsvPaths[11])
modelRgbLab = load_model(modelCsvPaths[12])
modelRgbYcbcr = load_model(modelCsvPaths[13])
modelHsv = load_model(modelCsvPaths[14])
modelHsvLab = load_model(modelCsvPaths[15])
modelHsvYcbcr = load_model(modelCsvPaths[16])
modelLab = load_model(modelCsvPaths[17])
modelLabYcbcr = load_model(modelCsvPaths[18])
modelYCbCr = load_model(modelCsvPaths[19])





#modelRgb = load_model('model_rgb')
#modelRgbHsv = load_model('model_rgb_hsv')
#modelRgbLab = load_model('model_rgb_lab')
#modelRgbYcbcr = load_model('model_rgb_ycbcr')
#modelHsv = load_model('model_hsv')
#modelHsvLab = load_model('model_hsv_lab')
#modelHsvYcbcr = load_model('model_hsv_ycbcr')
#modelLab = load_model('model_lab')
#modelLabYcbcr = load_model('model_lab_ycbcr')
#modelYCbCr = load_model('model_YCrCb')

#load inpaint model
#modelIpRgb = load_model('model_inpaint_rgb')
#modelIpRgbHsv = load_model('model_inpaint_rgb_hsv')
#modelIpRgbLab = load_model('model_inpaint_rgb_lab')
#modelIpRgbYcbcr = load_model('model_inpaint_rgb_ycbcr')
#modelIpHsv = load_model('model_inpaint_hsv')
#modelIpHsvLab = load_model('model_inpaint_hsv_lab')
#modelIpHsvYcbcr = load_model('model_inpaint_hsv_ycbcr')
#modelIpLab = load_model('model_inpaint_lab')
#modelIpLabYcbcr = load_model('model_inpaint_lab_ycbcr')
#modelIpYCbCr = load_model('model_inpaint_YCrCb')

print('All model loaded successfully..')
print('Set default graph..')
global graph
graph = tf.get_default_graph()
print('Default graph set succesfully.')
print('App is ready to use.')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/estimasi')
def estimasi():
    return render_template('estimasi.html')

@app.route('/latih')
def latih():
    return render_template('latih_cust.html')

@app.route('/latih_master')
def latih_master():
    return render_template('latih_master.html')

@app.route('/record')
def record():
    return render_template('record.html')

@app.route('/informasi')
def informasi():
    return render_template('informasi.html')

@app.route('/sign')
def sign():
    return render_template('sign.html')

@app.route('/users')
def users():
    return render_template('user_data.html')


@app.route('/cek_user', methods=['GET', 'POST'])
def cek_user():

    return jsonify({'user': currentUserName, 'id':currentUserId, 'auth' : currentAuth})

@app.route('/create_project', methods=['GET', 'POST'])
def create_project():
    project = request.form['project']
    filename = "static/project_record/project.csv"
    data = load_csv(filename)
    data = np.array(data)
    data = data.flatten()

    for i in data:
        if(i == project):
            print('project already exist!')
            return jsonify({'error' : 'error'})


    project_path_np = py_path + '/static/model/' + project + '/no_inpaint/'
    project_path_ip = py_path + '/static/model/' + project + '/inpaint/'

    arrModels = ['rgb', 'rgbhsv', 'rgblab', 'rgbycbcr', 'hsv', 'hsvlab', 'hsvycbcr', 'lab', 'labycbcr', 'ycbcr']
    create_dir(project_path_np)    
    create_dir(project_path_ip)

    for i in range(len(arrModels)):
        create_dir(project_path_np + arrModels[i]) 
        create_dir(project_path_ip + arrModels[i])  

    with open('static/project_record/project.csv', mode='a', newline='') as score:
        score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        score_writer.writerow([project, currentUserName])

    projCsv = load_csv(projectPath)
    projId = len(projCsv) + 1
    with open(projectPath, mode='a', newline='') as score:
        score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        score_writer.writerow([projId, project, userId, currentUserName])
    return jsonify({'sukses' : 'sukses', 'id' : projId})

@app.route('/sign_out', methods=['GET', 'POST'])
def sign_out():
    global currentUserId
    currentUserId = 0
    global currentUserName
    currentUserName = ''
    global currentAuth
    currentAuth = 0

    return jsonify({'user': currentUserName, 'id': currentUserId})

@app.route('/sign_in', methods=['GET', 'POST'])
def sign_in():
    filename = "static/project_record/user.csv"
    data = load_csv(filename)

    nama = request.form['name']
    password = request.form['pass']

    encr = hashlib.md5(password.encode()) 
    encr = encr.hexdigest()

    status = 'gagal'
    for i in data:
        if nama == i[1] and encr == i[2]:
            global currentUserId
            global currentUserName
            global currentAuth
            currentUserId = i[0]
            currentUserName = i[1]
            currentAuth = i[3]
            status = 'sukses' 


    return jsonify({'sukses': status, 'nama': currentUserName, 'id': currentUserId})

@app.route('/sign_up', methods=['GET', 'POST'])
def sign_up():
    filename = "static/project_record/user.csv"
    data = load_csv(filename)

    nama = request.form['name']
    password = request.form['pass']

    encr = hashlib.md5(password.encode()) 
    encr = encr.hexdigest()
    idUser = len(data) + 1
    status = 'sukses'
    for i in data:
        if nama == i[1]:
               status = 'gagal'   

    if status == 'sukses':
        with open('static/project_record/user.csv', mode='a', newline='') as score:
                score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                score_writer.writerow([idUser, nama, encr, 1])
    
    return jsonify({'sukses': status})

@app.route('/get_user', methods=['GET', 'POST'])
def get_user():
    filename = "static/project_record/user.csv"
    data = load_csv(filename)
    data = np.array(data).tolist()


    return jsonify({'sukses': 'status', 'user': data})

@app.route('/delete_user', methods=['GET', 'POST'])
def delete_user():

    idUser = request.form['id']

    filename = "static/project_record/user.csv"
    data = load_csv(filename)
    found = 0

    for i in range(len(data)):
        if(data[i][0] == idUser ):
            print(data[i])
            print(i)
            print('found')
            position = i
            found = 1

    if found == 0:
        print('not found')
    else:
        data.remove(data[position])

        with open(filename, mode='w', newline='') as score:
            score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            print('delete csv ok')
            for i in range(len(data)):
                score_writer.writerow([data[i][0], data[i][1], data[i][2]])    

    return jsonify({'sukses': found})

@app.route('/show_project', methods=['GET', 'POST'])
def show_project():
    filename = "static/project_record/project.csv"
    data = load_csv(filename)

    projectData = load_csv(projectPath)
    userData = load_csv(userPath)

    projUserData = []

    for i in projectData:
        if i[3] == currentUserName:
            row = []
            row.append(i[0])
            row.append(i[1])        
            row.append(i[3])
            projUserData.append(row)  
        
        
             

   
    return jsonify({'sukses' : 'sukses', 'project' : data, 'projects' : projUserData})

@app.route('/saveTrResult', methods=['GET', 'POST'])
def saveTrResult():

    filename = "static/project_record/project.csv"
    data = load_csv(filename)

    project = request.form['project']
    print(project)
    #projectId = request.form['id']
    colorSpace = request.form['color']
    loss = request.form['loss']
    mae = request.form['mae']
    valLoss = request.form['valLoss']
    valMae = request.form['valMae']
    status = request.form['mode']
    if(request.form['mode']== 'yes'):
        ipPath = '_inpaint'
    else:
        ipPath = ''

    if(colorSpace == "rgb"):
        file = 'static/project_record/rgb'+ipPath+'.csv'
    elif(colorSpace == "hsv"):
        file = 'static/project_record/hsv'+ipPath+'.csv'
    elif(colorSpace == "lab"):
        file = 'static/project_record/lab'+ipPath+'.csv'
    elif(colorSpace == "ycbcr"):
        file = 'static/project_record/ycbcr'+ipPath+'.csv'
    elif(colorSpace == "rgb_hsv"):
       file = 'static/project_record/rgb_hsv'+ipPath+'.csv'
    elif(colorSpace == "hsv_lab"):
       file = 'static/project_record/hsv_lab'+ipPath+'.csv'  
    elif(colorSpace == "lab_ycbcr"):
       file = 'static/project_record/lab_ycbcr'+ipPath+'.csv'  
    elif(colorSpace == "rgb_lab"):
       file = 'static/project_record/rgb_lab'+ipPath+'.csv'     
    elif(colorSpace == "hsv_ycbcr"):
       file = 'static/project_record/hsv_ycbcr'+ipPath+'.csv'   
    elif(colorSpace == "rgb_ycbcr"):
       file = 'static/project_record/rgb_ycbcr'+ipPath+'.csv'   

    '''files = ['rgb', 'rgb_hsv', 'rgb_lab', 'rgb_ycbcr', 'hsv', 'hsv_lab', 'hsv_ycbcr', 'lab', 'lab_ycbcr', 'ycbcr']
    warna = ['RGB', 'RGB_HSV', 'RGB_LAB', 'RGB_YCbCr', 'HSV', 'HSV_LAB', 'HSV_YCbCr', 'LAB', 'LAB_YCbCr', 'YCbCr']  
    indexCOlor = 100

    for i in range(len(files)):
        if colorSpace == files[i]:
            indexCOlor = i
    nilai = load_csv(akurasiPath)
    nilaiId = len(nilai) + 1'''

    with open(file, mode='a', newline='') as score:
        score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        score_writer.writerow([project, loss, mae, valLoss, valMae, ts, currentUserName])


    '''with open(akurasiPath, mode='a', newline='') as score:
        score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        score_writer.writerow([nilaiId, userId, indexCOlor, projectId, status, loss, mae, valLoss, valMae, ts])'''
    #save
    print("saving model")
    print(dirToSave)
    with open(dirToSave + ".json", "w") as json_file:
        json_file.write(modelJsonToSave)
    print(dirToSave)
    modelToSave.save_weights(dirToSave + ".h5")
    print("finished : model saved")  

    return jsonify({'sukses' : 'sukses', 'project' : project, 'color' : colorSpace})

def image_converter(data, img):
    convert = 'cv2.COLOR_BGR2'
    
    if len(data) == 2:
        color1 = data[0]
        color2 = data[1]
        if(color1 == 'YCbCr'):
            color1 = 'YCrCb'
        if(color2 == 'YCbCr'):
            color2 = 'YCrCb'

        image = cv2.cvtColor(img, eval(convert + color1))
        image2 = cv2.cvtColor(img, eval(convert + color2))

        image = cv2.resize(image, (34, 34))
        image2 = cv2.resize(image2, (34, 34))
            
        image = np.array(image)
        image2 = np.array(image2)
            
        image = np.concatenate((image, image2), axis=2)
    else:
        if(data[0] == 'YcbCr'):
            color1 = 'YCrCb'
        image = cv2.cvtColor(img, eval(convert + color1))
        image = cv2.resize(image, (34, 34))

    return image


@app.route('/do_train_custom', methods=['GET','POST'])
def do_train_custom():


    try:
        data = request.form.getlist("x[]")
        ip = request.form["ip"]
        project = request.form["project"]

        print(data)
        print(ip)

        for i in data:
            print(i)

        print(len(data))

        if len(data) == 2:
            colorName = data[0] + ' ' + data[1]
            color1 = data[0].lower()
            color2 = data[1].lower()
            color = color1 + '_' + color2
            if color1 == 'ycbcr':
                color1 = 'ycbcr'
            elif color2 == 'ycbcr':
                color2 = 'ycbcr'
            colorSpace = color1 + '_' + color2
            dirColor = colorSpace.replace('_', '')
        else:
            colorName = data[0]
            color1 = data[0].lower()
            color = color1
            if color1 == 'ycbcr':
                color1 = 'ycbcr'
            colorSpace = color1
            dirColor = colorSpace

        path = ""
        modelSavePath = ""

        now = datetime.now()

        timestamp = datetime.timestamp(now)
        dt_object = datetime.fromtimestamp(timestamp)
        date = str(dt_object).replace(' ','_')
        global ts
        ts = date
        date = date.replace(':','-')
        

        if(ip=="true"):
            directory = "static/train_data/room_inpaint/"
            arrayDirRoom = os.listdir(directory)
            fileName = "model_inpaint_" + colorSpace + '_' + date
            dirName = '/inpaint/'
            inpaint = 'yes'
            print('ip true')
            
        else:
            directory = "static/train_data/room/"
            arrayDirRoom = os.listdir(directory)
            fileName = "model_" + colorSpace + '_' + date
            dirName = '/no_inpaint/'
            inpaint = 'no'
            print('ip false')

        print(colorSpace)

        all_images = []
        for i in range(len(arrayDirRoom)):
            img = cv2.imread(directory + arrayDirRoom[i], cv2.IMREAD_UNCHANGED)
            image = image_converter(data, img)
            all_images.append(image)

        dirName = py_path + '/static/model' + '/'+ project +  dirName +  dirColor #directory to save model
        
        shape = all_images[0].shape

        X_data = np.array(all_images)
        X_data = X_data.astype(np.floating)
        X_data /= 255
        print("X : ", X_data.shape)

        data = load_csv("static/train_data/data_target.csv")
        data = np.array(data)

        dataset = list()
        
        for i in range(len(arrayDirRoom)):
            for j in range(len(data)):
                if( data[j, 0] == os.path.splitext(arrayDirRoom[i])[0]):
                    dataset.append(data[j, 1:5])
                    
        Y_data = np.array(dataset)
        print("Y : ", Y_data.shape)
        #shuffle data
        X_data, Y_data = shuffle(X_data, Y_data, random_state=2)
        #split data
        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.1, random_state=2)
        
        datagen = ImageDataGenerator(rotation_range=40,
                                zoom_range=0.2, 
                                fill_mode='nearest')
    
        
        #number of epoch and batch size
        EPOCHS = 200
        BS = 29
        with graph.as_default():
            model = Sequential()
            model.add(Conv2D(32, 
                    kernel_size=(3,3),
                    strides=(2, 2),
                    activation='relu',
                    input_shape=shape))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            model.add(Flatten())
            model.add(Dense(90, activation='relu'))
            model.add(Dropout(0.03))
            model.add(Dense(3))
            model.add(LeakyReLU(alpha=0.7))
            model.compile(loss="mean_squared_error",
                optimizer=Nadam(),
                metrics=['mae'])
            history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size= BS), 
                        validation_data=(X_test, Y_test), 
                        
                        epochs = EPOCHS)
        
        
        
        
        model_json = model.to_json()

        global modelJsonToSave
        modelJsonToSave = model_json
        global dirToSave 
        dirToSave = str( dirName + '/'+fileName) 
        global modelToSave
        modelToSave = model

        
        loss = history.history['loss'][-1]
        mae = history.history['mean_absolute_error'][-1]
        valLoss = history.history['val_loss'][-1]
        valMae = history.history['val_mean_absolute_error'][-1]

        #loss = truncate(loss, 9)
        #mae = truncate(mae, 9)
        #valLoss = truncate(valLoss, 9)
        #valMae = truncate(valMae, 9)
        
        
        print('sukses')

        return jsonify({'success' : 'Sukses', 
                        'loss' : loss, 
                        'mae' : mae,
                        'valLoss' : valLoss,
                        'valMae' : valMae, 
                        'color' : colorName,
                        'inpaint' : inpaint,
                        'warna' : color})
    except:
        return jsonify({'gagal' : 'gagal', 
                        'loss' : str(1000000), 
                        'mae' : str(1000000),
                        'valLoss' : str(1000000),
                        'valMae' : str(1000000), 
                        'color' : str(1000000),
                        'inpaint' : str(1000000)
                        })
   

@app.route('/do_train', methods=['GET','POST'])
def do_train():
    print('mulai')
    colorSpace = request.form['color']
    inpaint = request.form['inpaint']
    project = request.form['project']

    path = ""
    modelSavePath = ""

    now = datetime.now()

    timestamp = datetime.timestamp(now)
    dt_object = datetime.fromtimestamp(timestamp)
    date = str(dt_object).replace(' ','_')
    global ts
    ts = date
    date = date.replace(':','-')
    

    if(inpaint=="yes"):
        directory = "static/train_data/room_inpaint/"
        arrayDirRoom = os.listdir(directory)
        fileName = "model_inpaint_" + colorSpace + '_' + date
        dirName = '/inpaint/'
        
    else:
        directory = "static/train_data/room/"
        arrayDirRoom = os.listdir(directory)
        fileName = "model_" + colorSpace + '_' + date
        dirName = '/no_inpaint/'
    
    all_images = []
    for i in range(len(arrayDirRoom)):
        img = cv2.imread(directory + arrayDirRoom[i], cv2.IMREAD_UNCHANGED)
        if(colorSpace == "rgb"):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif(colorSpace == "hsv"):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif(colorSpace == "lab"):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif(colorSpace == "ycbcr"):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif(colorSpace == "rgb_hsv"):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif(colorSpace == "hsv_lab"):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            image2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif(colorSpace == "lab_ycbcr"):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
            image2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif(colorSpace == "rgb_lab"):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif(colorSpace == "hsv_ycbcr"):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            image2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif(colorSpace == "rgb_ycbcr"):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            image2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        if(len(colorSpace) > 5):
            colorName = colorSpace.replace('_', ' ')
            dirColor = colorSpace.replace('_', '') + '/'
            colorName = colorName.upper()
            image = cv2.resize(image, (34, 34))
            image2 = cv2.resize(image2, (34, 34))
            
            image = np.array(image)
            image2 = np.array(image2)
            
            image = np.concatenate((image, image2), axis=2)
        else:
            colorName = colorSpace.upper()
            dirColor = colorSpace + '/'
            image = cv2.resize(image, (34, 34))
            image = np.array(image)
            
        all_images.append(image)
    print(dirColor)

    dirName = py_path + '/static/model' + '/'+ project +  dirName + dirColor #directory to save model
    
    shape = all_images[0].shape
    
    X_data = np.array(all_images)
    X_data = X_data.astype(np.floating)
    X_data /= 255
    print("X : ", X_data.shape)
    
    data = load_csv("static/train_data/data_target.csv")
    data = np.array(data)

    dataset = list()
    
    for i in range(len(arrayDirRoom)):
        for j in range(len(data)):
            if( data[j, 0] == os.path.splitext(arrayDirRoom[i])[0]):
                dataset.append(data[j, 1:5])
                
    Y_data = np.array(dataset)
    print("Y : ", Y_data.shape)
    #shuffle data
    X_data, Y_data = shuffle(X_data, Y_data, random_state=2)
    #split data
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.1, random_state=2)
    
    datagen = ImageDataGenerator(rotation_range=40,
                             zoom_range=0.2, 
                             fill_mode='nearest')
   
    
    #number of epoch and batch size
    EPOCHS = 200
    BS = 29
    with graph.as_default():
         model = Sequential()
         model.add(Conv2D(32, 
                 kernel_size=(3,3),
                 strides=(2, 2),
                 activation='relu',
                 input_shape=shape))
         model.add(Conv2D(32, (3, 3), activation='relu'))
         model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
         model.add(Flatten())
         model.add(Dense(90, activation='relu'))
         model.add(Dropout(0.03))
         model.add(Dense(3))
         model.add(LeakyReLU(alpha=0.7))
         model.compile(loss="mean_squared_error",
              optimizer=Nadam(),
              metrics=['mae'])
         history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size= BS), 
                    validation_data=(X_test, Y_test), 
                    
                    epochs = EPOCHS)
    
    
    
    
    model_json = model.to_json()

    global modelJsonToSave
    modelJsonToSave = model_json
    global dirToSave 
    dirToSave = str( dirName + fileName) 
    global modelToSave
    modelToSave = model

    
    loss = history.history['loss'][-1]
    mae = history.history['mean_absolute_error'][-1]
    valLoss = history.history['val_loss'][-1]
    valMae = history.history['val_mean_absolute_error'][-1]
    
    
    print(colorSpace)
    print(inpaint)
    print(project)
    print(dirName)
    print('sukses')
    return jsonify({'success' : 'Sukses', 
                    'loss' : loss, 
                    'mae' : mae,
                    'valLoss' : valLoss,
                    'valMae' : valMae, 
                    'color' : colorName,
                    'inpaint' : inpaint
                    })

@app.route('/estimate', methods=['GET','POST'])
def estimate():
    #saving uploaded image
    target = os.path.join(APP_ROOT, 'static/')
    
    img = request.files['file']
    
    print("{} is the file name".format(img.filename))
    filename = img.filename
    destination = "/".join([target, 'daun.jpg'])
    destination_inpaint = "/".join([target, 'daun_inpainted.jpg'])
    print("Accept incoming file:", filename)
    print("Save it to:", destination)
    img.save(destination)
    
    
    #retrieve image
    inputImage = cv2.imread('static/daun.jpg',  cv2.IMREAD_UNCHANGED)

    #create inpainted version of input image
    inpaintedImg = inpaint_image(inputImage)
    cv2.imwrite(destination_inpaint, inpaintedImg)
    inpaintedInputImage = cv2.imread('static/daun_inpainted.jpg',  cv2.IMREAD_UNCHANGED)
    

    #preprocess non inpainted image into various color spaces
    rgb = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LAB)
    ycbcr = cv2.cvtColor(inputImage, cv2.COLOR_BGR2YCrCb)

    #preprocess inpainted image into various color spaces
    rgbIp = cv2.cvtColor(inpaintedInputImage, cv2.COLOR_BGR2RGB)
    hsvIp = cv2.cvtColor(inpaintedInputImage, cv2.COLOR_BGR2HSV)
    labIp = cv2.cvtColor(inpaintedInputImage, cv2.COLOR_BGR2LAB)
    ycbcrIp = cv2.cvtColor(inpaintedInputImage, cv2.COLOR_BGR2YCrCb)

    #resize non inpainted images
    rgb = np.array(cv2.resize(rgb, ( 34, 34)))
    hsv = np.array(cv2.resize(hsv, ( 34, 34)))
    lab = np.array(cv2.resize(lab, ( 34, 34)))
    ycbcr = np.array(cv2.resize(ycbcr, ( 34, 34)))

    #resize  inpainted images
    rgbIp = np.array(cv2.resize(rgbIp, ( 34, 34)))
    hsvIp = np.array(cv2.resize(hsvIp, ( 34, 34)))
    labIp = np.array(cv2.resize(labIp, ( 34, 34)))
    ycbcrIp = np.array(cv2.resize(ycbcrIp, ( 34, 34)))

    #create non inpainted 6 channel image 
    rgbHsv = np.concatenate((rgb, hsv), axis=2)
    rgbLab = np.concatenate((rgb, lab), axis=2)
    rgbYcbcr = np.concatenate((rgb, ycbcr), axis=2)
    hsvLab = np.concatenate((hsv, lab), axis=2)
    hsvYcbcr = np.concatenate((rgb, ycbcr), axis=2)
    labYcbcr = np.concatenate((rgb, ycbcr), axis=2)

    #create inpainted 6 channel image 
    rgbHsvIp = np.concatenate((rgbIp, hsvIp), axis=2)
    rgbLabIp = np.concatenate((rgbIp, labIp), axis=2)
    rgbYcbcrIp = np.concatenate((rgbIp, ycbcrIp), axis=2)
    hsvLabIp = np.concatenate((hsvIp, labIp), axis=2)
    hsvYcbcrIp = np.concatenate((rgbIp, ycbcrIp), axis=2)
    labYcbcrIp = np.concatenate((rgbIp, ycbcrIp), axis=2)

    #make 4 dimensional non inpainted image data 
    imRgb = image_appender(rgb)
    imRgbHsv = image_appender(rgbHsv)
    imRgbLab = image_appender(rgbLab)
    imRgbYcbcr = image_appender(rgbYcbcr)
    imHsv = image_appender(hsv)
    imHsvLab = image_appender(hsvLab)
    imHsvYcbcr = image_appender(hsvYcbcr)
    imLab = image_appender(lab)
    imLabYcbcr = image_appender(labYcbcr)
    imYcbcr = image_appender(ycbcr)

    #make 4 dimensional inpainted image data 
    imRgbIp = image_appender(rgbIp)
    imRgbHsvIp = image_appender(rgbHsvIp)
    imRgbLabIp = image_appender(rgbLabIp)
    imRgbYcbcrIp = image_appender(rgbYcbcrIp)
    imHsvIp = image_appender(hsvIp)
    imHsvLabIp = image_appender(hsvLabIp)
    imHsvYcbcrIp = image_appender(hsvYcbcrIp)
    imLabIp = image_appender(labIp)
    imLabYcbcrIp = image_appender(labYcbcrIp)
    imYcbcrIp = image_appender(ycbcrIp)
    

    #get pigment estimation data 
    with graph.as_default():
        #non inpainted image
        npRgb = modelRgb.predict(imRgb)
        npRgbHsv = modelRgbHsv.predict(imRgbHsv)
        npRgbLab = modelRgbLab.predict(imRgbLab)
        npRgbYcbcr = modelRgbYcbcr.predict(imRgbYcbcr)
        npHsv = modelHsv.predict(imHsv)
        npHsvLab = modelHsvLab.predict(imHsvLab)
        npHsvYcbcr = modelHsvYcbcr.predict(imHsvYcbcr)
        npLab = modelLab.predict(imLab)
        npLabYcbcr = modelLabYcbcr.predict(imLabYcbcr)
        npYCbCr = modelYCbCr.predict(imYcbcr)

        #inpainted image
        ipRgb = modelIpRgb.predict(imRgbIp)
        ipRgbHsv = modelIpRgbHsv.predict(imRgbHsvIp)
        ipRgbLab = modelIpRgbLab.predict(imRgbLabIp)
        ipRgbYcbcr = modelIpRgbYcbcr.predict(imRgbYcbcrIp)
        ipHsv = modelIpHsv.predict(imHsvIp)
        ipHsvLab = modelIpHsvLab.predict(imHsvLabIp)
        ipHsvYcbcr = modelIpHsvYcbcr.predict(imHsvYcbcrIp)
        ipLab = modelIpLab.predict(imLabIp)
        ipLabYcbcr = modelIpLabYcbcr.predict(imLabYcbcrIp)
        ipYCbCr = modelIpYCbCr.predict(imYcbcrIp)

    #split 3 pigments result into variables

    #non inpainted image
    npRgb1, npRgb2, npRgb3 = res_splitter(npRgb)
    npRgbHsv1, npRgbHsv2, npRgbHsv3 = res_splitter(npRgbHsv)
    npRgbLab1, npRgbLab2, npRgbLab3 = res_splitter(npRgbLab)
    npRgbYcbcr1, npRgbYcbcr2, npRgbYcbcr3 = res_splitter(npRgbYcbcr)
    npHsv1, npHsv2, npHsv3 = res_splitter(npHsv)
    npHsvLab1, npHsvLab2, npHsvLab3 = res_splitter(npHsvLab)
    npHsvYcbcr1, npHsvYcbcr2, npHsvYcbcr3 = res_splitter(npHsvYcbcr)
    npLab1, npLab2, npLab3 = res_splitter(npLab)   
    npLabYcbcr1, npLabYcbcr2, npLabYcbcr3 = res_splitter(npLabYcbcr) 
    npYCbCr1, npYCbCr2, npYCbCr3 = res_splitter(npYCbCr)

    #inpainted image
    ipRgb1, ipRgb2, ipRgb3 = res_splitter(ipRgb)
    ipRgbHsv1, ipRgbHsv2, ipRgbHsv3 = res_splitter(ipRgbHsv)
    ipRgbLab1, ipRgbLab2, ipRgbLab3 = res_splitter(ipRgbLab)
    ipRgbYcbcr1, ipRgbYcbcr2, ipRgbYcbcr3 = res_splitter(ipRgbYcbcr)
    ipHsv1, ipHsv2, ipHsv3 = res_splitter(ipHsv)
    ipHsvLab1, ipHsvLab2, ipHsvLab3 = res_splitter(ipHsvLab)
    ipHsvYcbcr1, ipHsvYcbcr2, ipHsvYcbcr3 = res_splitter(ipHsvYcbcr)
    ipLab1, ipLab2, ipLab3 = res_splitter(ipLab)   
    ipLabYcbcr1, ipLabYcbcr2, ipLabYcbcr3 = res_splitter(ipLabYcbcr) 
    ipYCbCr1, ipYCbCr2, ipYCbCr3 = res_splitter(ipYCbCr)
    
    
    return jsonify({'success' : 'sukses!',        
                    'npRgbAnto': npRgb1, 'npRgbKaro': npRgb2, 'npRgbKlor': npRgb3,
                    'npRgbHsvAnto': npRgbHsv1, 'npRgbHsvKaro': npRgbHsv2, 'npRgbHsvKlor': npRgbHsv3,
                    'npRgbLabAnto': npRgbLab1, 'npRgbLabKaro': npRgbLab2, 'npRgbLabKlor': npRgbLab3,
                    'npRgbYcbcrAnto': npRgbYcbcr1, 'npRgbYcbcrKaro': npRgbYcbcr2, 'npRgbYcbcrKlor': npRgbYcbcr3,
                    'npHsvAnto': npHsv1, 'npHsvKaro': npHsv2, 'npHsvKlor': npHsv3,
                    'npHsvLabAnto': npHsvLab1, 'npHsvLabKaro': npHsvLab2, 'npHsvLabKlor': npHsvLab3,
                    'npHsvYcbcrAnto': npHsvYcbcr1, 'npHsvYcbcrKaro': npHsvYcbcr2, 'npHsvYcbcrKlor': npHsvYcbcr3,
                    'npLabAnto': npLab1, 'npLabKaro': npLab2, 'npLabKlor': npLab3,
                    'npLabYcbcrAnto': npLabYcbcr1, 'npLabYcbcrKaro': npLabYcbcr2, 'npLabYcbcrKlor': npLabYcbcr3,
                    'npYCbCrAnto': npYCbCr1, 'npYCbCrKaro': npYCbCr2, 'npYCbCrKlor': npYCbCr3,
                    'ipRgbAnto': ipRgb1, 'ipRgbKaro': ipRgb2, 'ipRgbKlor': ipRgb3,
                    'ipRgbHsvAnto': ipRgbHsv1, 'ipRgbHsvKaro': ipRgbHsv2, 'ipRgbHsvKlor': ipRgbHsv3,
                    'ipRgbLabAnto': ipRgbLab1, 'ipRgbLabKaro': ipRgbLab2, 'ipRgbLabKlor': ipRgbLab3,
                    'ipRgbYcbcrAnto': ipRgbYcbcr1, 'ipRgbYcbcrKaro': ipRgbYcbcr2, 'ipRgbYcbcrKlor': ipRgbYcbcr3,
                    'ipHsvAnto': ipHsv1, 'ipHsvKaro': ipHsv2, 'ipHsvKlor': ipHsv3,
                    'ipHsvLabAnto': ipHsvLab1, 'ipHsvLabKaro': ipHsvLab2, 'ipHsvLabKlor': ipHsvLab3,
                    'ipHsvYcbcrAnto': ipHsvYcbcr1, 'ipHsvYcbcrKaro': ipHsvYcbcr2, 'ipHsvYcbcrKlor': ipHsvYcbcr3,
                    'ipLabAnto': ipLab1, 'ipLabKaro': ipLab2, 'ipLabKlor': ipLab3,
                    'ipLabYcbcrAnto': ipLabYcbcr1, 'ipLabYcbcrKaro': ipLabYcbcr2, 'ipLabYcbcrKlor': ipLabYcbcr3,
                    'ipYCbCrAnto': ipYCbCr1, 'ipYCbCrKaro': ipYCbCr2, 'ipYCbCrKlor': ipYCbCr3})

@app.route('/show_scores', methods=['GET', 'POST'])
def show_scores():
    project = request.form['project']
    files = ['rgb', 'rgb_hsv', 'rgb_lab', 'rgb_ycbcr', 'hsv', 'hsv_lab', 'hsv_ycbcr', 'lab', 'lab_ycbcr', 'ycbcr']
    path = "static/project_record/"

    #load all each color scoring
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

    #precisely insert list of project's scores
    rgb = np.array(load_records(allRgb, project)).tolist()
    rgbHsv = np.array(load_records(allRgbHsv, project)).tolist()
    rgbLab = np.array(load_records(allRgbLab, project)).tolist()
    rgbYcb = np.array(load_records(allRgbYcb, project)).tolist()
    hsv = np.array(load_records(allHsv, project)).tolist()
    hsvLab = np.array(load_records(allHsvLab, project)).tolist()
    hsvYcb = np.array(load_records(allHsvYcb, project)).tolist()
    lab = np.array(load_records(allLab, project)).tolist()
    labYcbcr = np.array(load_records(allLabYcb, project)).tolist()
    ycbcr = np.array(load_records(allYcb, project)).tolist()

    rgbIp = np.array(load_records(allRgbIp, project)).tolist()
    rgbHsvIp = np.array(load_records(allRgbHsvIp, project)).tolist()
    rgbLabIp = np.array(load_records(allRgbLabIp, project)).tolist()
    rgbYcbIp = np.array(load_records(allRgbYcbIp, project)).tolist()
    hsvIp = np.array(load_records(allHsvIp, project)).tolist()
    hsvLabIp = np.array(load_records(allHsvLabIp, project)).tolist()
    hsvYcbIp = np.array(load_records(allHsvYcbIp, project)).tolist()
    labIp = np.array(load_records(allLabIp, project)).tolist()
    labYcbcrIp = np.array(load_records(allLabYcbIp, project)).tolist()
    ycbcrIp = np.array(load_records(allYcbIp, project)).tolist()
    
    avRgb = get_avScore(sort_score(rgb))
    print('rgb average : ', avRgb[1])
    sortRgb = sort_score(rgb)
    for i in sortRgb:
        print(i)
    avRgbHsv = get_avScore(sort_score(rgbHsv))
    avRgbLab = get_avScore(sort_score(rgbLab))
    avRgbYcb = get_avScore(sort_score(rgbYcb))
    avHsv = get_avScore(sort_score(hsv))
    avHsvLab = get_avScore(sort_score(hsvLab))
    avHsvYcb = get_avScore(sort_score(hsvYcb))
    avLab = get_avScore(sort_score(lab))
    avLabYcb = get_avScore(sort_score(labYcbcr))
    avYcbcr = get_avScore(sort_score(ycbcr))

    avRgbIp = get_avScore(sort_score(rgbIp))
    avRgbHsvIp = get_avScore(sort_score(rgbHsvIp))
    avRgbLabIp = get_avScore(sort_score(rgbLabIp))
    avRgbYcbIp = get_avScore(sort_score(rgbYcbIp))
    avHsvIp = get_avScore(sort_score(hsvIp))
    avHsvLabIp = get_avScore(sort_score(hsvLabIp))
    avHsvYcbIp = get_avScore(sort_score(hsvYcbIp))
    avLabIp = get_avScore(sort_score(labIp))
    avLabYcbIp = get_avScore(sort_score(labYcbcrIp))
    avYcbcrIp = get_avScore(sort_score(ycbcrIp))

    losses = [
        ['RGB', avRgb[0], avRgb[2]], ['RGB+HSV', avRgbHsv[0], avRgbHsv[2]], ['RGB+LAB', avRgbLab[0], avRgbLab[2]], ['RGB+YCbCr', avRgbYcb[0], avRgbYcb[2]], 
        ['HSV', avHsv[0], avHsv[2]], ['HSV+LAB', avHsvLab[0], avHsvLab[2]], ['HSV+YCbCr', avHsvYcb[0], avHsvYcb[2]], 
        ['LAB', avLab[0], avLab[2]], ['LAB+YCbCr', avLabYcb[0], avLabYcb[2]], ['YCbCr', avYcbcr[0], avYcbcr[2]]
        ] 

    lossesIp = [
        ['RGB', avRgbIp[0], avRgbIp[2]], ['RGB+HSV', avRgbHsvIp[0], avRgbHsvIp[2]], ['RGB+LAB', avRgbLabIp[0], avRgbLabIp[2]], ['RGB+YCbCr', avRgbYcbIp[0], avRgbYcbIp[2]], 
        ['HSV', avHsvIp[0], avHsvIp[2]], ['HSV+LAB', avHsvLabIp[0], avHsvLabIp[2]], ['HSV+YCbCr', avHsvYcbIp[0], avHsvYcbIp[2]], 
        ['LAB', avLabIp[0], avLabIp[2]], ['LAB+YCbCr', avLabYcbIp[0], avLabYcbIp[2]], ['YCbCr', avYcbcrIp[0], avYcbcrIp[2]]
        ] 

    mae = [
        ['RGB', avRgb[1], avRgb[3]], ['RGB+HSV', avRgbHsv[1], avRgbHsv[3]], ['RGB+LAB', avRgbLab[1], avRgbLab[3]], ['RGB+YCbCr', avRgbYcb[1], avRgbYcb[3]], 
        ['HSV', avHsv[1], avHsv[3]], ['HSV+LAB', avHsvLab[1], avHsvLab[3]], ['HSV+YCbCr', avHsvYcb[1], avHsvYcb[3]], 
        ['LAB', avLab[1], avLab[3]], ['LAB+YCbCr', avLabYcb[1], avLabYcb[3]], ['YCbCr', avYcbcr[1], avYcbcr[3]]
        ] 

    maeIp = [
        ['RGB', avRgbIp[1], avRgbIp[3]], ['RGB+HSV', avRgbHsvIp[1], avRgbHsvIp[3]], ['RGB+LAB', avRgbLabIp[1], avRgbLabIp[3]], ['RGB+YCbCr', avRgbYcbIp[1], avRgbYcbIp[3]], 
        ['HSV', avHsvIp[1], avHsvIp[3]], ['HSV+LAB', avHsvLabIp[1], avHsvLabIp[3]], ['HSV+YCbCr', avHsvYcbIp[1], avHsvYcbIp[3]], 
        ['LAB', avLabIp[1], avLabIp[3]], ['LAB+YCbCr', avLabYcbIp[1], avLabYcbIp[3]], ['YCbCr', avYcbcrIp[1], avYcbcrIp[3]]
        ] 

    imLoss = create_chart(losses, project, 'loss', '', 'Loss Not Inpainted')
    imLossIp = create_chart(lossesIp, project, 'loss', 'inpainted', 'Loss Inpainted')
    imMae = create_chart(mae, project, 'mae', '', 'Mae Not Inpainted')
    imMaeIp = create_chart(maeIp, project, 'mae', 'inpainted', 'Mae Inpainted')

    #create csv
    
    with open('static/project_record/'+ project +'skor.csv', mode='w', newline='') as score:
        score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        score_writer.writerow(['Project :', project])
        score_writer.writerow(['RGB'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        print('length rgb : ', len(rgb))
        for i in range(len(rgb)):
            score_writer.writerow([rgb[i][0], rgb[i][1], rgb[i][2], rgb[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['RGB + HSV'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(rgbHsv)):
            score_writer.writerow([rgbHsv[i][0], rgbHsv[i][1], rgbHsv[i][2], rgbHsv[i][3]])
            
        score_writer.writerow([])
        score_writer.writerow(['RGB + LAB'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(rgbLab)):
            score_writer.writerow([rgbLab[i][0], rgbLab[i][1], rgbLab[i][2], rgbLab[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['RGB + YCbCr'])
        for i in range(len(rgbYcb)):
            score_writer.writerow([rgbYcb[i][0], rgbYcb[i][1], rgbYcb[i][2], rgbYcb[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['HSV'])
        for i in range(len(hsv)):
            score_writer.writerow([hsv[i][0], hsv[i][1], hsv[i][2], hsv[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['HSV + LAB'])
        for i in range(len(hsvLab)):
            score_writer.writerow([hsvLab[i][0], hsvLab[i][1], hsvLab[i][2], hsvLab[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['HSV + YCbCr'])
        for i in range(len(hsvYcb)):
            score_writer.writerow([hsvYcb[i][0], hsvYcb[i][1], hsvYcb[i][2], hsvYcb[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['LAB'])
        for i in range(len(lab)):
            score_writer.writerow([lab[i][0], lab[i][1], lab[i][2], lab[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['LAB + YCbCr'])
        for i in range(len(labYcbcr)):
            score_writer.writerow([labYcbcr[i][0], labYcbcr[i][1], labYcbcr[i][2], labYcbcr[i][3]])

        score_writer.writerow([])
        score_writer.writerow([' YCbCr'])
        for i in range(len(ycbcr)):
            score_writer.writerow([ycbcr[i][0], ycbcr[i][1], ycbcr[i][2], ycbcr[i][3]])
        score_writer.writerow([])

        score_writer.writerow(['RGB Inpaint'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        print('length rgbip : ', len(rgbIp))
        for i in range(len(rgbIp)):
            score_writer.writerow([rgbIp[i][0], rgbIp[i][1], rgbIp[i][2], rgbIp[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['RGB + HSV Inpaint'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(rgbHsvIp)):
            score_writer.writerow([rgbHsvIp[i][0], rgbHsvIp[i][1], rgbHsvIp[i][2], rgbHsvIp[i][3]])
            
        score_writer.writerow([])
        score_writer.writerow(['RGB + LAB Inpaint'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(rgbLabIp)):
            score_writer.writerow([rgbLabIp[i][0], rgbLabIp[i][1], rgbLabIp[i][2], rgbLabIp[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['RGB + YCbCr Inpaint'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(rgbYcbIp)):
            score_writer.writerow([rgbYcbIp[i][0], rgbYcbIp[i][1], rgbYcbIp[i][2], rgbYcbIp[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['HSV Inpaint'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(hsvIp)):
            score_writer.writerow([hsvIp[i][0], hsvIp[i][1], hsvIp[i][2], hsvIp[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['HSV + LAB Inpaint'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(hsvLabIp)):
            score_writer.writerow([hsvLabIp[i][0], hsvLabIp[i][1], hsvLabIp[i][2], hsvLabIp[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['HSV + YCbCr Inpaint'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(hsvYcbIp)):
            score_writer.writerow([hsvYcbIp[i][0], hsvYcbIp[i][1], hsvYcbIp[i][2], hsvYcbIp[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['LAB  Inpaint'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(labIp)):
            score_writer.writerow([labIp[i][0], labIp[i][1], labIp[i][2], labIp[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['LAB + YCbCr Inpaint'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(labYcbcrIp)):
            score_writer.writerow([labYcbcrIp[i][0], labYcbcrIp[i][1], labYcbcrIp[i][2], labYcbcrIp[i][3]])

        score_writer.writerow([])
        score_writer.writerow([' YCbCr Inpaint'])
        score_writer.writerow(['Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(ycbcrIp)):
            score_writer.writerow([ycbcrIp[i][0], ycbcrIp[i][1], ycbcrIp[i][2], ycbcrIp[i][3]])

        score_writer.writerow([])
        score_writer.writerow(['Rata-rata Skor'])
        score_writer.writerow(['Tanpa Inpaint'])       
        score_writer.writerow(['Ruang Warna','Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(losses)):
            score_writer.writerow([losses[i][0],losses[i][1], mae[i][1], losses[i][2], mae[i][2]])

        score_writer.writerow([])
        score_writer.writerow(['Dengan Inpaint'])       
        score_writer.writerow(['Ruang Warna','Train Loss', 'Train Mae', 'Val Loss', 'Val Mae'])
        for i in range(len(losses)):
            score_writer.writerow([lossesIp[i][0],lossesIp[i][1], maeIp[i][1], lossesIp[i][2], maeIp[i][2]])

    print('length rgbip : ', len(rgbIp))

    return jsonify({'sukses' : 'sukses', 
                    'project' : project,
                    'rgb' : rgb,
                    'rgbHsv' : rgbHsv,
                    'rgbLab' : rgbLab,
                    'rgbYcbcr' : rgbYcb,
                    'hsv' : hsv,
                    'hsvLab' : hsvLab,
                    'hsvYcbcr' : hsvYcb,
                    'lab' : lab,
                    'labYcbcr' : labYcbcr,
                    'ycbcr' : ycbcr,
                    'rgbIp' : rgbIp,
                    'rgbHsvIp' : rgbHsvIp,
                    'rgbLabIp' : rgbLabIp,
                    'rgbYcbcrIp' : rgbYcbIp,
                    'hsvIp' : hsvIp,
                    'hsvLabIp' : hsvLabIp,
                    'hsvYcbcrIp' : hsvYcbIp,
                    'labIp' : labIp,
                    'labYcbcrIp' : labYcbcrIp,
                    'ycbcrIp' : ycbcrIp,
                    'avRgb' : avRgb,
                    'avRgbHsv' : avRgbHsv,
                    'avRgbLab' : avRgbLab,
                    'avRgbYcb' : avRgbYcb,
                    'avHsv' : avHsv,
                    'avHsvLab' : avHsvLab,
                    'avHsvYcb' : avHsvYcb,
                    'avLab' : avLab,
                    'avLabYcb' : avLabYcb,
                    'avYcbcr' : avYcbcr,
                    'avRgbIp' : avRgbIp,
                    'avRgbHsvIp' : avRgbHsvIp,
                    'avRgbLabIp' : avRgbLabIp,
                    'avRgbYcbIp' : avRgbYcbIp,
                    'avHsvIp' : avHsvIp,
                    'avHsvLabIp' : avHsvLabIp,
                    'avHsvYcbIp' : avHsvYcbIp,
                    'avLabIp' : avLabIp,
                    'avLabYcbIp' : avLabYcbIp,
                    'avYcbcrIp' : avYcbcrIp,
                    'imLoss' :imLoss, 'imLossIp' : imLossIp, 'imMae' : imMae, 'imMaeIp' : imMaeIp,
                    })

@app.route('/delete_model', methods=['GET', 'POST'])
def delete_model():
    color = request.form['color']
    time = request.form['tStamp']
    project = request.form['project']
    mode = request.form['mode']

    modelPath = 'static/model/' + project 
    if( 'ip' in color):
        color = color.replace('ip', '')

    if(mode== 'yes'):
        ipPath = '_inpaint'
        modelPath = modelPath + '/inpaint/'+color+'/model_inpaint_'

    else:
        ipPath = ''
        modelPath = modelPath + '/no_inpaint/'+color+'/model_'  

    if len(color) > 3 and color != 'ycbcr':
        color = color[:3] + '_' + color[3:]

    modelPath = modelPath + color + '_' + time
    modelPath = modelPath.replace(':', '-')
    print(modelPath)
    
    print(color)
    file = ''
    
    if(color == "rgb"):
        file = 'static/project_record/rgb'+ipPath+'.csv'
    elif(color == "hsv"):
        file = 'static/project_record/hsv'+ipPath+'.csv'
    elif(color == "lab"):
        file = 'static/project_record/lab'+ipPath+'.csv'
    elif(color == "ycbcr"):
        file = 'static/project_record/ycbcr'+ipPath+'.csv'
    elif(color == "rgb_hsv"):
       file = 'static/project_record/rgb_hsv'+ipPath+'.csv'
    elif(color == "hsv_lab"):
       file = 'static/project_record/hsv_lab'+ipPath+'.csv'  
    elif(color == "lab_ycbcr"):
       file = 'static/project_record/lab_ycbcr'+ipPath+'.csv'  
    elif(color == "rgb_lab"):
       file = 'static/project_record/rgb_lab'+ipPath+'.csv'     
    elif(color == "hsv_ycbcr"):
       file = 'static/project_record/hsv_ycbcr'+ipPath+'.csv'   
    elif(color == "rgb_ycbcr"):
       file = 'static/project_record/rgb_ycbcr'+ipPath+'.csv'  

    record = load_csv(file);
    found = 0

    for i in range(len(record)):
        if(record[i][0] == project and record[i][5] == time):
            print(record[i])
            print(i)
            print('found')
            position = i
            found = 1

    if found == 0:
        print('not found')
        print(project)
        print(time)


    record.remove(record[position])
    print(record)

    with open(file, mode='w', newline='') as score:
        score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        print('delete csv ok')
        for i in range(len(record)):
            score_writer.writerow([record[i][0], record[i][1], record[i][2], record[i][3], record[i][4], record[i][5], record[i][6]])

    try:
        os.remove(modelPath + ".h5")
        os.remove(modelPath + ".json")
        print('file deleted')
    except:
        print('file not found')
    return jsonify({'success' : 'Sukses'})


def getProject(data):

    project = list()

    for i in range(len(data)):
        project.append(data[i][0])

    project = list(set(project))

    return project

@app.route('/get_selected_model', methods=['GET', 'POST'])
def get_selected_model():
    path = 'static/project_record/model_select.csv'

    selected = np.array(load_csv(path)).tolist()

    akurasiData = load_csv(akurasiPath)
    modelData = load_csv(modelSelPath)
    projData = load_csv(projectPath)
    warnaData = load_csv(warnaPath)

    selectedModel = []

    for i in range(len(modelData)):
        
        for j in akurasiData:
            
            mod = 'model_'
            warna = ''
            proj = ''
            inpaint = ''
            ts = ''

            if modelData[i][1] == j[0]:
                row = []
                for k in warnaData:
                    if j[2] == k[0]:
                        warna = k[1]
                for l in projData:
                    if j[3] == l[0]:
                        proj = l[1]
                inpaint = j[4]
                if inpaint == 'yes':
                    mod = mod + 'inpaint_' + warna.lower()
                else:
                    mod = mod + warna.lower()
                ts = j[9]
                row.append(warna)
                row.append(inpaint)
                row.append(proj)
                row.append(mod)                
                row.append(ts)
                selectedModel.append(row)
        

    return jsonify({ 'selected' : selected, 'selModel' : selectedModel })
    
@app.route('/pilih_model', methods=['GET', 'POST'])
def pilh_model():
    project = request.form['project']
    color = request.form['color']
    inpaint = request.form['inpaint']

    warna = color.lower()
    print(warna)
    judulWarna = color.replace('_', ' ')
    color = color.replace('_', '')

    path = 'static/model/' + project + '/'
    if inpaint == 'yes':
        path = path + 'inpaint/' + color + '/'
        status = 'inpaint_'
        st = '_inpaint'
    else :
        path = path + 'no_inpaint/' + color + '/'
        status = ''
        st = ''


    print(path)

    files = ['rgb', 'rgb_hsv', 'rgb_lab', 'rgb_ycbcr', 'hsv', 'hsv_lab', 'hsv_ycbcr', 'lab', 'lab_ycbcr', 'ycbcr']
    pathScores = "static/project_record/"


    models = list()

    for file in os.listdir(path):
        if file.endswith(".h5"):
            file = file.replace('.h5', '')
            models.append(file)
    creator = []
    for i in files:
        if i == warna:
            data = load_csv(pathScores + i + st + '.csv')
            for j in data:
                judulModel = 'model_' + status + warna + '_' + j[5].replace(':', '-')
               
                for k in models:
                    if judulModel == k and project == j[0]:
                        print('yes')
                        creator.append(j[6])
                    


    print(creator)




    return jsonify({ 'sukses' : 'sukses', 'models' : models, 'project' : project,'color' : judulWarna, 'creator' : creator})

@app.route('/fix_pilih_model', methods=['GET', 'POST'])
def fix_pilih_model():
    project = request.form['project']
    color = request.form['color']
    inpaint = request.form['inpaint']
    model = request.form['model']

    

    import re
    pos = [m.start() for m in re.finditer(r"_",model)]

    namaModel = model[:pos[2]]

    ts = model[pos[2] + 1:] #timestamp
    

    path = 'static/model/' + project + '/'
    selModelColor = color
    color = color.replace('_', '')
     #timestamp
    if re.search('inpaint', model):
        if len(color) > 5:
            namaModel = model[:pos[3]]
            ts = model[pos[3] + 1:]
        else:
            namaModel = model[:pos[2]]
            ts = model[pos[2] + 1:]
    else:
        if len(color) > 5:
            namaModel = model[:pos[2]]
            ts = model[pos[2] + 1:]
        else:
            namaModel = model[:pos[1]]
            ts = model[pos[1] + 1:]

    clock = ts.replace('-', ':') 
        

    if inpaint == 'yes':
        path = path + 'inpaint/' + color + '/' + model

        if(color == "rgb"):
            global modelIpRgb
            with graph.as_default():
                modelIpRgb = load_model(path)
        elif(color == "hsv"):
            global modelIpHsv
            with graph.as_default():
                modelIpHsv = load_model(path)            
        elif(color == "lab"):
            global modelIpLab
            with graph.as_default():
                modelIpLab = load_model(path)
        elif(color == "ycbcr"):
            global modelIpYCbCr
            with graph.as_default():
                modelIpYCbCr = load_model(path)            
        elif(color == "rgb_hsv"):
            global modelIpRgbHsv
            with graph.as_default():
                modelIpRgbHsv = load_model(path)
        elif(color == "hsv_lab"):
            global modelIpHsvLab
            with graph.as_default():
                modelIpHsvLab = load_model(path)
        elif(color == "lab_ycbcr"):
            global modelIpLabYcbcr
            with graph.as_default():
                modelIpLabYcbcr = load_model(path)          
        elif(color == "rgb_lab"):
            global modelIpRgbLab
            with graph.as_default():
                modelIpRgbLab = load_model(path)   
        elif(color == "hsv_ycbcr"):
            global modelIpHsvYcbcr
            with graph.as_default():
                modelIpHsvYcbcr = load_model(path)
        elif(color == "rgb_ycbcr"):
            global modelIpRgbYcbcr
            with graph.as_default():
                modelIpRgbYcbcr = load_model(path)
    else:
        path = path + 'no_inpaint/' + color + '/' + model
        if(color == "rgb"):
            global modelRgb
            with graph.as_default():
                modelRgb = load_model(path)
        elif(color == "hsv"):
            global modelHsv
            with graph.as_default():
                modelHsv = load_model(path)            
        elif(color == "lab"):
            global modelLab
            with graph.as_default():
                modelLab = load_model(path)
        elif(color == "ycbcr"):
            global modelYCbCr
            with graph.as_default():
                modelYCbCr = load_model(path)            
        elif(color == "rgb_hsv"):
            global modelRgbHsv
            with graph.as_default():
                modelRgbHsv = load_model(path)
        elif(color == "hsv_lab"):
            global modelHsvLab
            with graph.as_default():
                modelHsvLab = load_model(path)
        elif(color == "lab_ycbcr"):
            global modelLabYcbcr
            with graph.as_default():
                modelLabYcbcr = load_model(path)          
        elif(color == "rgb_lab"):
            global modelRgbLab
            with graph.as_default():
                modelRgbLab = load_model(path)   
        elif(color == "hsv_ycbcr"):
            global modelHsvYcbcr
            with graph.as_default():
                modelHsvYcbcr = load_model(path)
        elif(color == "rgb_ycbcr"):
            global modelRgbYcbcr
            with graph.as_default():
                modelRgbYcbcr = load_model(path)
        

    print('mulai')

    print('ok')
    path_csv = 'static/project_record/model_select.csv'
    

    selected = np.array(load_csv(path_csv))

    modelSel = list()

    for i in selected:
        if i[0] == selModelColor and i[1] == inpaint:
            modelSel.append(np.array([selModelColor, inpaint, project, namaModel, clock]).flatten())
        else:
            modelSel.append(np.array(i).flatten())


    with open(path_csv, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(np.array(modelSel))
      
    print(ts)
    print(color)
    print(model)
    print(project)
    print(inpaint)
    print(path)


    return jsonify({'sukses' : 'sukses'})



@app.route('/get_available_project', methods=['GET', 'POST'])
def get_available_project():
    files = ['rgb', 'rgb_hsv', 'rgb_lab', 'rgb_ycbcr', 'hsv', 'hsv_lab', 'hsv_ycbcr', 'lab', 'lab_ycbcr', 'ycbcr']
    path = "static/project_record/"

    
    color = request.form['color']
    inpaint = request.form['inpaint']
    
    print(color)
    print(inpaint)

    path_inpaint = ''

    if inpaint == 'yes':
        path_inpaint = '_inpaint'

    csvPath = path + color + path_inpaint + '.csv'

    print(csvPath)
    available = getProject(np.array(load_csv(csvPath)))

    print(available)

    akurasiData = load_csv(akurasiPath)
    projData = load_csv(projectPath)
    userData = load_csv(userPath)
    warnaData = load_csv(warnaPath)

    projCreator = []

    for i in available:
        for j in projData:
            if i == j[1]:
                projCreator.append(j[3])

    print(projData)
    print(projCreator)
    return jsonify({'sukses' : 'sukses', 
                    'available' : available,
                    'creator' : projCreator
                    })




@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(port=4500, threaded=True, debug=True)





