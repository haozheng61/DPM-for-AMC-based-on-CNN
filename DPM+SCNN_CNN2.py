import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import pickle


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

snrs = ""
mods = ""
test_idx = ""
lbl = ""
ttrate = 0.7


def gendata(fp):
    global snrs, mods, test_idx, lbl
    Xd = pickle.load(open(fp, 'rb'), encoding='bytes')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
    X = np.vstack(X)

    np.random.seed(2019)
    n_examples = X.shape[0]
    n_train = int(n_examples * ttrate)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    X_train = X[train_idx]
    X_test = X[test_idx]

    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    return (X_train, X_test, Y_train, Y_test)

def FeatureReverse(a,step):
    def Reverse(a,s):
        b = a
        i = 1
        while (1):
            p = i * (i + 1) * s / 2
            p = int(p)
            b = np.r_[b, np.c_[a[:, (len(a[0]) - p):], a[:, :(len(a[0]) - p)]]]
            i = i + 1
            if p + (i + 1) * s > len(a[0]):
                break
        return b
    ab = np.array([])
    for i in range(a.shape[0]):
        ab = np.append(ab,Reverse(a[i],step))
    ab = np.array(ab).reshape((a.shape[0], int(ab.shape[0]/(a.shape[0]*a.shape[2])), a.shape[2]))
    return ab

xtrain1, xtest1, ytrain1, ytest1 = gendata("D:\\RML2016.10a_dict.pkl")

Rstep = 4
trainbatch = int(xtrain1.shape[0]/1000)   #Distributed processing to effectively increase computing speed
testbatch = int(xtest1.shape[0]/1000)
x_train1 = FeatureReverse(xtrain1[:trainbatch],Rstep)
x_test1 = FeatureReverse(xtest1[:testbatch],Rstep)
for i in range(99):
    x_train1 = np.r_[x_train1,FeatureReverse(xtrain1[(i+1)*trainbatch:(i+2)*trainbatch],Rstep)]
    x_test1 = np.r_[x_test1,FeatureReverse(xtest1[(i+1)*testbatch:(i+2)*testbatch],Rstep)]
print('FR 10%')
for i in range(100):
    x_train1 = np.r_[x_train1,FeatureReverse(xtrain1[i*trainbatch+trainbatch*100:(i+1)*trainbatch+trainbatch*100],Rstep)]
    x_test1 = np.r_[x_test1,FeatureReverse(xtest1[i*testbatch+testbatch*100:(i+1)*testbatch+testbatch*100],Rstep)]
print('FR 20%')
for i in range(100):
    x_train1 = np.r_[x_train1,FeatureReverse(xtrain1[i*trainbatch+trainbatch*100*2:(i+1)*trainbatch+trainbatch*100*2],Rstep)]
    x_test1 = np.r_[x_test1,FeatureReverse(xtest1[i*testbatch+testbatch*100*2:(i+1)*testbatch+testbatch*100*2],Rstep)]
print('FR 30%')
for i in range(100):
    x_train1 = np.r_[x_train1,FeatureReverse(xtrain1[i*trainbatch+trainbatch*100*3:(i+1)*trainbatch+trainbatch*100*3],Rstep)]
    x_test1 = np.r_[x_test1,FeatureReverse(xtest1[i*testbatch+testbatch*100*3:(i+1)*testbatch+testbatch*100*3],Rstep)]
print('FR 40%')
for i in range(100):
    x_train1 = np.r_[x_train1,FeatureReverse(xtrain1[i*trainbatch+trainbatch*100*4:(i+1)*trainbatch+trainbatch*100*4],Rstep)]
    x_test1 = np.r_[x_test1,FeatureReverse(xtest1[i*testbatch+testbatch*100*4:(i+1)*testbatch+testbatch*100*4],Rstep)]
print('FR 50%')
for i in range(100):
    x_train1 = np.r_[x_train1,FeatureReverse(xtrain1[i*trainbatch+trainbatch*100*5:(i+1)*trainbatch+trainbatch*100*5],Rstep)]
    x_test1 = np.r_[x_test1,FeatureReverse(xtest1[i*testbatch+testbatch*100*5:(i+1)*testbatch+testbatch*100*5],Rstep)]
print('FR 60%')
for i in range(100):
    x_train1 = np.r_[x_train1,FeatureReverse(xtrain1[i*trainbatch+trainbatch*100*6:(i+1)*trainbatch+trainbatch*100*6],Rstep)]
    x_test1 = np.r_[x_test1,FeatureReverse(xtest1[i*testbatch+testbatch*100*6:(i+1)*testbatch+testbatch*100*6],Rstep)]
print('FR 70%')
for i in range(100):
    x_train1 = np.r_[x_train1,FeatureReverse(xtrain1[i*trainbatch+trainbatch*100*7:(i+1)*trainbatch+trainbatch*100*7],Rstep)]
    x_test1 = np.r_[x_test1,FeatureReverse(xtest1[i*testbatch+testbatch*100*7:(i+1)*testbatch+testbatch*100*7],Rstep)]
print('FR 80%')
for i in range(100):
    x_train1 = np.r_[x_train1,FeatureReverse(xtrain1[i*trainbatch+trainbatch*100*8:(i+1)*trainbatch+trainbatch*100*8],Rstep)]
    x_test1 = np.r_[x_test1,FeatureReverse(xtest1[i*testbatch+testbatch*100*8:(i+1)*testbatch+testbatch*100*8],Rstep)]
print('FR 90%')
for i in range(100):
    x_train1 = np.r_[x_train1,FeatureReverse(xtrain1[i*trainbatch+trainbatch*100*9:(i+1)*trainbatch+trainbatch*100*9],Rstep)]
    x_test1 = np.r_[x_test1,FeatureReverse(xtest1[i*testbatch+testbatch*100*9:(i+1)*testbatch+testbatch*100*9],Rstep)]
print('FR 100%')
print(x_train1.shape)
print(x_test1.shape)
print('FR complete')

from numpy import linalg as la
def amp_phase(a):
    b = np.zeros(a.shape)
    n = int(a.shape[1]/2)
    for i in range(n):
        x_tran = a[:,i*2,:] + 1j*a[:,i*2+1,:]
        b[:,i*2,:] = np.abs(x_tran)
        b[:,i*2+1,:] = np.arctan2(a[:,i*2,:], a[:,i*2+1,:]) / np.pi
    return b
def norm(a,b):
    for i in range(a.shape[0]):
        norm_amp = 1 / la.norm(a[i, 0, :], 2)
        b[i,:,:] = b[i,:,:] * norm_amp
        for j in range(int(a.shape[1]/2)):
            a[i,j*2,:] = a[i,j*2,:] * norm_amp
    return  a,b
x_train1a = amp_phase(x_train1)
x_test1a = amp_phase(x_test1)
x_train1a,x_train1 = norm(x_train1a,x_train1)
x_test1a,x_test1 = norm(x_test1a,x_test1)
x_train1 = x_train1.reshape(x_train1.shape[0],x_train1.shape[1],x_train1.shape[2],1)
x_train1a = x_train1a.reshape(x_train1a.shape[0],x_train1a.shape[1],x_train1a.shape[2],1)
x_test1 = x_test1.reshape(x_test1.shape[0],x_test1.shape[1],x_test1.shape[2],1)
x_test1a = x_test1a.reshape(x_test1a.shape[0],x_test1a.shape[1],x_test1a.shape[2],1)

xtrain = np.concatenate((x_train1,x_train1a),axis=3)
xtest = np.concatenate((x_test1,x_test1a),axis=3)

def channeltrans(a):
    b = np.zeros(a.shape)
    for i in range(int(a.shape[1]/2)):
        b[:,i*2:i*2+2,:,:] = np.swapaxes(a[:,i*2:i*2+2,:,:],1,3)
    return b

x_train1 = channeltrans(xtrain)
x_test1 = channeltrans(xtest)

X_train = x_train1
X_test = x_test1

Y_train = ytrain1
Y_test = ytest1

NB_CLASSES=len(Y_train[0])

print("--" * 50)
print("Training data:", X_train.shape)
print("Training labels:", Y_train.shape)
print("Testing data", X_test.shape)
print("Testing labels", Y_test.shape)
print("--" * 50)


def getFontColor(value):
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"

def getConfusionMatrixPlot(true_labels, predicted_labels):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm, 2)
    print(cm)

    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=1)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    # add genres as ticks
    alphabet = mods
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    return plt


from keras import layers
from keras import Sequential, layers
from keras.layers import Reshape, Conv2D, Dense, Activation, Dropout, Flatten, ZeroPadding2D
from keras import optimizers
from keras import models
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, ZeroPadding2D, GlobalAveragePooling2D, Activation, Average, \
    Dropout, GaussianNoise, LSTM, GlobalMaxPooling2D, Add, BatchNormalization, Concatenate
from keras import Model
from keras import regularizers

reduce_lr = ReduceLROnPlateau(monitor='loss', patience=3, mode='auto', factor=0.01)
adam0 = optimizers.Adam(lr=0.0005)

in_shp=X_train.shape[1:]

from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import *

def SCNN(input_shape, classes):
    x_input = Input(input_shape)
    x1 = ZeroPadding2D(padding=((0, 0), (0, 4)))(x_input)
    x1 = Conv2D(84, (2, 8), strides=(2, 1), padding='valid')(x1)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)
    x2 = ZeroPadding2D(padding=((0, 2), (0, 0)))(x_input)
    x2 = Conv2D(84, (4, 4), strides=(2, 1), padding='valid')(x2)
    x2 = Activation('relu')(x2)
    x2 = BatchNormalization()(x2)
    x = Concatenate(axis=3)([x1, x2])
    x = ZeroPadding2D(padding=((0, 0), (0, 3)))(x)
    def Res_block(y):
        shortcut_unit = y
        y1 = Conv2D(84, (1, 8), padding='same')(y)
        y1 = Activation('relu')(y1)
        y1 = BatchNormalization()(y1)
        y1 = Dropout(0.5)(y1)
        y2 = Conv2D(84, (2, 3), padding='same')(y)
        y2 = Activation('relu')(y2)
        y2 = BatchNormalization()(y2)
        y2 = Dropout(0.5)(y2)
        y = Concatenate(axis=3)([y1, y2])
        y = Add()([shortcut_unit, y])
        return y
    x = Res_block(x)
    x = Res_block(x)
    x = Conv2D(168, (2, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(168, (1, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, (2, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (2, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(512, (1, 3), padding='same')(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = GaussianNoise(0.01)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes)(x)
    x_out = Activation('softmax')(x)
    model = Model(inputs=x_input, outputs=x_out, name='SCNN')
    return model

def CNN2(input_shape, classes):
    x_input = Input(input_shape)
    x = Conv2D(256,(4,4) , strides= (2,1) , padding='valid')(x_input)
    x = Activation('relu')(x)
    x = Conv2D(256,(3,3) , padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.6)(x)
    x = Conv2D(80,(3,3) , padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(80,(3,3) , padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.6)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(classes)(x)
    x_out = Activation('softmax')(x)
    model = Model(inputs=x_input, outputs=x_out, name='CNN2')
    return model

model = SCNN(in_shp, NB_CLASSES)
# model = CNN2(in_shp, NB_CLASSES)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=adam0, metrics=['accuracy'])

batch_size = 256
epochs = 100
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks=[reduce_lr]
                    )
model.save('D:\\DPM+SCNN.h5')
# model.save('D:\\DPM+CNN2.h5')
print(history.history['loss'])
print('')
print(history.history['val_loss'])
plt.figure()
plt.title('Training performance')
plt.plot(history.history['loss'], label='train loss+error')
plt.show()
plt.plot(history.history['val_loss'], label='val_error')
plt.show()
acc = {}


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


for snr in snrs:

    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_SNRs = list(test_SNRs)
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)

    conf = np.zeros([len(mods), len(mods)])
    confnorm = np.zeros([len(mods), len(mods)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(mods)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

    plot_confusion_matrix(confnorm, labels=mods, title="ConvNet Confusion Matrix (SNR=%d)" % (snr))
    plt.show()

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor

    acc[snr] = 1.0 * cor / (cor + ncor)
    print('snr:', snr)
    print('acc:', acc[snr])

plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("DP+SCNN Classification Accuracy on RadioML 2016.10 Alpha")
# plt.title("DP+CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
plt.yticks(np.linspace(0, 1, 6))
plt.show()