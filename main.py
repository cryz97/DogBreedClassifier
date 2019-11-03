import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session() # Limpia sesiones activas de KERAS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

trainingDataSetPath = './assets/training'
validationDataSetPath = './assets/validation'

# Parametros

epochsIterations= 20
pixelsWidth, pixelsHeight = 100, 100
batchSize = 32  # Numero de imagenes a procesar en cada iteración
steps = 1000  # Numero de veces que se procesará la información en cada periodo
stepsValidation = 300 # Iteraciones de validación al termino de cada periodo
convolutionFilter1 = 32  # Despues de cada convolucion se tendrá una profundidad de 32
convolutionFilter2 = 64  # " "           "   "                   "   "             64
filterSize1 = (3, 3)  # Tamaño del kernel que recorre las imagenes en la primera convolucion
filterSize2 = (2, 2)  # " "      " "         " "         " "            segunda   "  "
maxPoolSize = (2, 2)  # Tamaño del kernel en el max pooling
classesQuantity = 120  # Cantidad de clases que tenemos (para el ejemplo se usarán 3, tenemos 120 totales)
learningRate = 0.0004  # Ajuste de la CNN para acercarse a la solución óptima

# Pre procesmiento de imagenes

dataGenTraining = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

dataGenValidation = ImageDataGenerator(
    rescale=1./255
)

trainingImage = dataGenTraining.flow_from_directory(
    trainingDataSetPath,
    target_size=(pixelsHeight, pixelsWidth),
    batch_size=batchSize,
    class_mode='categorical'
)

validationImage = dataGenValidation.flow_from_directory(
    validationDataSetPath,
    target_size=(pixelsHeight,pixelsWidth),
    batch_size=batchSize,
    class_mode='categorical'
)

# CNN

CNN = Sequential()

# Definir la convolucion
CNN.add(
    Convolution2D(
        convolutionFilter1,

        filterSize1,
        padding='same',
        input_shape=(pixelsHeight, pixelsWidth , 3),
        activation='relu')  # relu funcion de activacion
)

CNN.add(
    MaxPooling2D(pool_size=maxPoolSize)
)

CNN.add(
    Convolution2D(
        convolutionFilter2,
        filterSize2,
        padding='same',
        activation='relu'))

CNN.add(
    MaxPooling2D(pool_size=maxPoolSize)
)

CNN.add(Flatten())
CNN.add(Dense(256,  # 256 neuronas
              activation='relu'))
CNN.add(Dropout(0.5)) # apagar 50% de neuronas en cada step (caminos alternos a soluciones)

CNN.add(Dense(classesQuantity, activation='softmax')) #3 neuronas, clasifica

CNN.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=learningRate),
            metrics=['accuracy'])


CNN.fit_generator(
    trainingImage,
    steps_per_epoch=steps,
    epochs=epochsIterations,
    validation_data=validationImage,
    validation_steps=stepsValidation)

dirModel = './model/'
if not os.path.exists(dirModel):
  os.mkdir(dirModel)
CNN.save('./model/Model.h5')
CNN.save_weights('./model/Weights.h5')
