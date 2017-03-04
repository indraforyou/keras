from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import RecurDataMem, Input, Dense, TimeDistributed, InputLayer
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

# inp = RecurDataMem(time_dim=10, input_dim=(64,64,1))
# out = TimeDistributed(Dense(32))(inp)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

inp = Input(shape=(28,28,1))
out = RecurDataMem(time_dim=4)(inp)

model = Model(input=inp, output=out)
model.summary()

func = K.function(inputs=[inp], outputs=[out], updates=model.updates)

plt.ion()

fig, axis = plt.subplots(nrows=1, ncols=1)
# for ax in axis:
# 	hnd = plt.imshow(stiched_fig(0))
hnd = plt.imshow(np.hstack([X_train[0],X_train[1],X_train[2],X_train[3]]))


plt.draw()
plt.pause(00001)
for idx in range(100):
	data = X_train[idx][..., np.newaxis]
	# output = model.predict(data[np.newaxis, ...])
	output = func([data[np.newaxis, ...]])
	# print output
	# print output[0].shape
	output = np.squeeze(output[0])
	print output.shape
	print output[0].shape

	out_img = np.hstack([output[0],output[1],output[2],output[3]])
	print out_img.shape
	
	hnd.set_data(out_img)

	plt.draw()
	plt.pause(00001)



# model = Sequential()
# model.add(InputLayer(input_shape=(64,64,1)))
# model.add(RecurDataMem(time_dim=10))
# model.summary()
