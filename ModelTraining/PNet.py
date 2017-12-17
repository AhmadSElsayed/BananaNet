import keras
from keras.models import Sequential
from keras.layers import Conv2d, MaxPooling2D, Activation, Fork
import os
import DatasetManager

# Hyper Params
batch_size = 128
epochs = 120
save_dir = os.path.join(os.getcwd(), 'saved_models')
models_name = 'PNet'

# The Data, Shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test), input_shape = DatasetManager.load_data('PNet')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# PNet Structure
model = Sequential()
# Conv1
model.add(Conv2d(filters = 10, strides = (1,1), kernel_size = (3,3), input_shape = input_shape))
model.add(Activation('prelu'))
# Pool1
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2))
#Conv2
model.add(Conv2d(filters = 16, strides = (1,1), kernel_size = (3,3)))
model.add(Activation('prelu'))
#Conv3
model.add(Conv2d(filters = 32, strides = (1,1), kernel_size = (3,3)))
model.add(Activation('prelu'))
#Fork ( Branch The Next Two Layers )
model.add(Fork('same', 2, 2))
#Conv4-1
model.add(Conv2d(filters = 2, strides = (1,1), kernel_size = (1,1)))
#Softmax
model.add(Activation('softmax'))
#Conv4-2
model.add(Conv2d(filters = 4, strides = (1,1), kernel_size = (1,1)))
#Adam
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#Compilation
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
# Training
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
# Evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('PNet.h5')