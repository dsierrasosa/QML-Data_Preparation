import warnings
warnings.filterwarnings('ignore')

import os
import json

import tensorflow as tf
import tensorflow_quantum as tfq
from cirq.contrib.svg import SVGCircuit
import keras
import tensorflow.keras.layers as KL

import sympy
import cirq
# import sympy
# import collections
import numpy as np
# import pandas as pd
import seaborn as sns
sns.set_style('dark')
import matplotlib.pyplot as plt

# import sklearn
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split

def Plus1Minus1(y_train, y_val, y_test):
    y_train = 2 * y_train - 1
    y_val = 2 * y_val - 1
    y_test = 2 * y_test - 1
    return y_train, y_val, y_test


def ShowShapes(x_train, x_test, x_val, y_train, y_val, y_test):
    print("make sure shapes are divisible by powers of 2")
    print("Classical Train:     ", x_train.shape[0])
    print("Classical Validation:", x_test.shape[0])
    print("Classical Test:      ", x_val.shape[0])
    print("-"*90)
    print("Train -> (-1,1):      ", y_train[:15])
    print("-"*90)
    print("Val -> (-1,1)  :      ", y_val[:15])
    print("-"*90)
    print("Test -> (-1,1) :      ", y_test[:15])


def PlotData(x, y):
    plt.figure()
    X = x
    plt.scatter(X[:, 0][y==0], X[:, 1][y==0], c='r', marker='^', edgecolors='k', label="train 0")
    plt.scatter(X[:, 0][y==1], X[:, 1][y==1], c='b', marker='o', edgecolors='k', label="train 1")
    plt.legend()
    plt.show()

def EncodeData(data):
    '''
    Perform amplitude encoding on each record of data. Row wise.
    '''
    ret = []
    for i in range(len(data)):
        ret.append(GetAngles1(np.array(data[i]), 2))
    return ret

def GetAngles1(x, n=2, verbose=False):
    '''
    Multi
    '''
    up = 0
    down = 0
    beta = []
    # print(x)
    # print()
    # print(n)
    for s in range (1, n+1):
        for j in range(1, n - s + 2):
            if verbose:
                print(str(s) + "->" + str(j))
            for l in range(0, 2 ** (s-1)):
                up = up + (x[(((2 * j - 1) * (2 ** (s - 1))) + l)] **2)
            for l in range(0, 2 ** (s)):
                down = down + (x[(((j - 1) * (2 ** (s))) + l)] **2)
            if s != n:
                beta.append(np.arcsin(np.sqrt(up) / np.sqrt(down)))
                beta.append(-1 * np.arcsin(np.sqrt(up) / np.sqrt(down)))
            else:
                beta.append(2 * np.arcsin(np.sqrt(up) / np.sqrt(down)))
            up = 0
            down = 0
    return np.fliplr([beta])[0]

def GenerateCircuits5(encoded, qubits, syms):
    '''
    Actual encoding using sympy
    '''
    
    circuits = []
    drawings = []
    
    for x in encoded:
        circuit = cirq.Circuit()
        
        circuit.append(cirq.ry(x[0])(cirq.GridQubit(0,0))**syms[0])
        circuit.append(cirq.CNOT(cirq.GridQubit(0,0), cirq.GridQubit(0,1)))
        circuit.append(cirq.ry(x[1])(cirq.GridQubit(0,1))**syms[1])
        circuit.append(cirq.CNOT(cirq.GridQubit(0,0), cirq.GridQubit(0,1)))
        circuit.append(cirq.ry(x[2])(cirq.GridQubit(0,1))**syms[2])
        ##
        circuit.append(cirq.X(cirq.GridQubit(0,0)))
        circuit.append(cirq.CNOT(cirq.GridQubit(0,0), cirq.GridQubit(0,1)))
        circuit.append(cirq.ry(x[3])(cirq.GridQubit(0,1))**syms[3])
        circuit.append(cirq.CNOT(cirq.GridQubit(0,0), cirq.GridQubit(0,1)))
        circuit.append(cirq.ry(x[4])(cirq.GridQubit(0,1))**syms[4])
        circuit.append(cirq.CNOT(cirq.GridQubit(0,0), cirq.GridQubit(0,1)))
        
        circuits.append(circuit)
        drawings.append(SVGCircuit(circuit))
        
    return tfq.convert_to_tensor(circuits), drawings

def one_qubit_unitary(bit, symbols):
    return cirq.Circuit(
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2])

def two_qubit_unitary(bits, symbols):
    '''
    Make a Cirq circuit that creates an arbitrary two qubit unitary.
    '''
    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(bits[0], symbols[0:3])
    circuit += one_qubit_unitary(bits[1], symbols[3:6])
    circuit += [cirq.ZZ(*bits)**symbols[7]]
    circuit += [cirq.YY(*bits)**symbols[8]]
    circuit += [cirq.XX(*bits)**symbols[9]]
    circuit += one_qubit_unitary(bits[0], symbols[9:12])
    circuit += one_qubit_unitary(bits[1], symbols[12:])
    return circuit

def two_qubit_pool(source_qubit, sink_qubit, symbols):
    '''
    Make a Cirq circuit to do a parameterized 'pooling' operation, which
    attempts to reduce entanglement down from two qubits to just one.
    '''
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, symbols[0:3])
    source_basis_selector = one_qubit_unitary(source_qubit, symbols[3:6])
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    pool_circuit.append(sink_basis_selector**-1)
    return pool_circuit

def quantum_conv_circuit(bits, symbols):
    circuit = cirq.Circuit()
    for first, second in zip(bits[0::2], bits[1::2]):
        circuit += two_qubit_unitary([first, second], symbols)
    for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
        circuit += two_qubit_unitary([first, second], symbols)
    return circuit

def quantum_pool_circuit(source_bits, sink_bits, symbols):
    '''
    A Quantum pool tries to learn to pool the relevant information from two
    qubits onto 1
    '''
    circuit = cirq.Circuit()
    for source, sink in zip(source_bits, sink_bits):
        circuit += two_qubit_pool(source, sink, symbols)
    return circuit


def multi_readout_model_circuit(qubits, symbols):
    """Make a model circuit with less quantum pool and conv operations."""
    model_circuit = cirq.Circuit()
    
    model_circuit += quantum_conv_circuit(qubits, symbols[0:15])
    model_circuit += quantum_pool_circuit(qubits[:4], qubits[4:], symbols[15:21])
    
    model_circuit += quantum_conv_circuit(qubits[4:], symbols[21:36])
    model_circuit += quantum_pool_circuit(qubits[4:6], qubits[6:], symbols[36:42])
    
    return model_circuit

@tf.function
def custom_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.map_fn(lambda x: 1.0 if x >= 0 else -1.0, y_pred)
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))


def modelEncodedQuantumCNN_MLP(amplitude_symbols, BATCH_SIZE):
    bit = cirq.GridQubit(0, 0)
    cluster_state_bits = cirq.GridQubit.rect(1, 8)
    ops = [-1.0 * cirq.Z(bit), cirq.X(bit) + 2.0 * cirq.Z(bit)] #[cirq.Z(bit) for bit in cluster_state_bits[4:]]
    learnable_gamma = tf.Variable(np.array([[1.,1.,1.,1.]]*BATCH_SIZE), dtype=float)

    cnn_syms_42 = sympy.symbols('qconv0:42')
    quantum_cnn = multi_readout_model_circuit(cluster_state_bits, cnn_syms_42)
    
    
    input_shape = tf.keras.Input(shape=(4,))
    circuit_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)

    dense_in = KL.Dense(43, activation='relu')(input_shape)

    datapoint_circuit_plus_model_circuit = tfq.layers.AddCircuit()(circuit_input, append=quantum_cnn)
    quantum_expectation = tfq.layers.Expectation()(
    datapoint_circuit_plus_model_circuit,
    symbol_names = amplitude_symbols + cnn_syms_42,
    symbol_values = tf.concat([dense_in, learnable_gamma], axis=1),
    operators=ops,)
    #   initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi))
    x = KL.Dense(32)(quantum_expectation)
    x = KL.Dense(1, activation='tanh')(x)

    model = tf.keras.Model(inputs=[input_shape,circuit_input], outputs=x)
    
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.02),
                    loss='mse',
                    metrics=[custom_accuracy])
    
    return model


def EncodedQuantumCNN_MLP(train=[], val=[], EPOCHS=1, BATCH_SIZE=2, qubits=cirq.GridQubit.rect(1,4)):
    assert (len(train) == 2), "Need x and y for train: [x_train, y_train]"
    assert (len(val) == 2), "Need x and y for val: [x_val, y_val]"

    amplitude_symbols = sympy.symbols("a b c d e")
    x_train_q, _ = GenerateCircuits5(EncodeData(train[0]), qubits, amplitude_symbols)
    x_val_q, _ = GenerateCircuits5(EncodeData(val[0]), qubits, amplitude_symbols)
    
    model = modelEncodedQuantumCNN_MLP(amplitude_symbols, BATCH_SIZE)

    history = model.fit(x=[train[0],x_train_q],
                            y=train[1],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=([val[0], x_val_q], val[1]))
    return model, history, amplitude_symbols

def plotloss(history_dict, loss='loss', val_loss='val_loss'):
    loss_values = history_dict[loss]
    val_loss_values = history_dict[val_loss]
    epochs = range(1, len(loss_values) + 1)
    
    plt.figure(figsize=(12,8))
    plt.plot(epochs, loss_values, 'r', label='Training Loss')
    plt.plot(epochs, val_loss_values, 'g', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


def plotacc(history_dict, acc='acc', val_acc='val_acc'):
    acc = history_dict[acc]
    val_acc = history_dict[val_acc]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,8))
    plt.plot(epochs, acc, 'r', label='Training Acc')
    plt.plot(epochs, val_acc, 'g', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

def main():
    space = 2.0
    x,y = skd.make_blobs(n_samples=3200, centers=2, n_features=4, random_state=6, center_box=(-space, space))

    x_train, x_test_, y_train, y_test_ = train_test_split(x, y, test_size=0.32, random_state=42)

    ## needs to be divisible by two for expectation layer
    x_train, y_train = x_train[0:2048], y_train[0:2048]

    x_test, x_val = x_test_[:int(len(x_test_)/2)], x_test_[int(len(x_test_)/2):]

    y_test, y_val = y_test_[:int(len(y_test_)/2)], y_test_[int(len(y_test_)/2):]

    y_train, y_val, y_test = Plus1Minus1(y_train, y_val, y_test)

    ShowShapes(x_train, x_test, x_val, y_train, y_val, y_test)

    __EPOCHS, __BATCH_SIZE = 50, 32
    run = str(space)
    prefix = "largelong"

    encodedQcnn1, encodedQcnnHistory1, amplitude_symbols = EncodedQuantumCNN_MLP([x_train, y_train], [x_val, y_val], EPOCHS=__EPOCHS, BATCH_SIZE=__BATCH_SIZE)

    x_test_q, _ = GenerateCircuits5(EncodeData(x_test), cirq.GridQubit.rect(1, 4), amplitude_symbols)

    history_dict = encodedQcnnHistory1
    plotloss(history_dict, 'loss', 'val_loss')
    plotacc(history_dict, 'acc', 'val_acc')

    evalulation = encodedQcnn1.evaluate(x_test, y_test)
    
    for i, ele in enumerate(evalulation):
        print("Model Evaluation:", str(encodedQcnn1.metrics_names[i]) + " ~ %.2f%%" % (ele * 100.0))

if __name__ == "__main__":
    main()
