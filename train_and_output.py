import tensorflow as tf  
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing.image import image as _image
from keras.applications import densenet  
from keras.models import Sequential, Model, load_model  
from keras.layers import Conv2D, MaxPooling2D  
from keras.layers import Activation, Dropout, Flatten, Dense  
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback  
from keras import regularizers  
from keras import backend as K 
from keras import losses
from keras import optimizers

path_base = 'dataset/v2'

img_width, img_height = 1024//2, 768//2
batch_size = 32

train_data_dir = path_base 

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.2,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data


#https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def F1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def create_model():

  model = Sequential()
  model.add(Conv2D(64, (5,5), input_shape = (img_height,img_width,3)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(5,5)))

  model.add(Conv2D(128, (4,4)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(4,4)))

  model.add(Conv2D(256, (3,3)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(3,3)))

  model.add(Flatten())

  model.add(Dense(3))
  model.add(Activation('sigmoid'))

  model.summary()
  model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc',F1])
  
  return model


model = create_model()

model_history=model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs =  5)

loss_, acc_, F1_ = model.evaluate_generator(validation_generator)

print('-- Trained Model Evaluation --')
print(f'Loss : {loss_} , Accuracy : {acc_}, F1 Score : {F1_}' )

#Save the model
model.save('ui_evaluator.h5')