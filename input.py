import numpy as np
import tkinter
from tkinter.filedialog import askopenfilename
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

height, width = 100, 100
dirModel = './model/Model.h5'
dirWeights = './model/Weights.h5'
dirModelTest = './model/testModels/testModel.h5'
dirWeightsTest = './model/testModels/testWeights.h5'
CNN = load_model(dirModel)
CNN.load_weights(dirWeights) #Carga de modelo y pesos

def ListClassesImages():
  return os.listdir( './assets/training')

def Predict(file):
  model = load_img(file, target_size=(height, width))
  model = img_to_array(model)
  model = np.expand_dims(model, axis=0)
  arrayResults = CNN.predict(model)
  firstResult = arrayResults[0]
  predictionResult = np.argmax(firstResult)
  directoryImagesList = ListClassesImages()
  print(predictionResult)
  print(directoryImagesList[predictionResult])

filename = askopenfilename() #Abre la ventana para seleccionar archivo
Predict(filename) #Manda dicho archivo a la CNN
