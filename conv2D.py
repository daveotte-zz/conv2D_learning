from keras.layers import Input, Dense, Conv2D, Conv1D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
from keras.models import load_model

#each sample is an image that is 28 x 28 with 1 B&W channel 
input_shape = Input(shape=(28, 28, 1))  

#The 2D Convolutional layer will learn 16 different filter
my_conv = Conv2D(filters=16, strides=1, kernel_size=(3,3), activation='relu', padding='same')
my_tensorflow_output_handle = my_conv(input_shape)

my_model = Model(input_shape, my_tensorflow_output_handle)
my_model.compile(optimizer='adadelta', loss='binary_crossentropy')

print "The kernel size is 3 x 3."
print "The output shape is: " + str(my_tensorflow_output_handle.shape) + " regardless of the kernel_size...why?"
print "I didn't expect the output shape to match the size of the images."

'''

(input_data, _), (x_test, _) = mnist.load_data()
input_data.reshape(60000,784)
input_data = np.reshape(input_data, (len(input_data), 28, 28, 1))  
#input shape (samples,row,col,channels)
output_data = np.arange(752640000).reshape(60000,28,28,16)
my_model.fit(input_data, output_data, epochs=5,batch_size=128)

my_model.save('/Users/daveotte/work/myConv2D.h5')


'''
