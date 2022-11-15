from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import Input, Dense, Subtract, TimeDistributed, Multiply, LSTM, Reshape, Concatenate, Maximum
import numpy as np

memory_size = 20  # 20, 40, 60
time_scale = 3600

inputLayer = Input(shape=(4, 1), name='input')
cpu = Dense(time_scale, name='CPU')(inputLayer)
cpu_lstm = LSTM(memory_size)(cpu)
memory = Dense(time_scale, name='Memory')(inputLayer)
memory_lstm = LSTM(memory_size)(memory)

mul1 = Multiply()([cpu_lstm, memory_lstm])

allocated0 = Dense(time_scale, name='Allocated')(mul1)
allocated = Reshape((time_scale, 1), input_shape=(time_scale,))(allocated0)
allocated_lstm = LSTM(memory_size)(allocated)
used0 = Dense(time_scale, name='Used')(mul1)
used = Reshape((time_scale, 1), input_shape=(time_scale,))(used0)
used_lstm = LSTM(memory_size)(used)

mul2 = Multiply()([allocated_lstm, used_lstm])

dense1 = Dense(time_scale, name='Dense1')(mul2)
dense2 = Dense(time_scale, name='Dense2')(mul2)

model = Model(inputs=[inputLayer], outputs=[dense1, dense2])
