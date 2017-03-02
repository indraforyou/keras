from keras.models import Sequential, Model
from keras.layers import RecurDataMem, Input, Dense, TimeDistributed


# inp = RecurDataMem(time_dim=10, input_dim=(64,64,1))
# out = TimeDistributed(Dense(32))(inp)

inp = Input(shape=(64,64,1))
out = RecurDataMem(time_dim=10)

model = Model(input=inp, output=out)
model.summary
