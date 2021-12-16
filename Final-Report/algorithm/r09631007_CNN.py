from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from service.analyticService.core.analyticCore.utils import XYdataGenerator, XdataGenerator
from service.analyticService.core.analyticCore.classificationBase import classification
from math import ceil

class r09631007_CNN(classification):
    def trainAlgo(self):
        self.model = Sequential()
        # Layer One
        self.model.add(Conv2D(self.param['layer_1_neuron'], (3, 3), input_shape = (64,64,3), activation = self.param['layer_1_activation']))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(self.param['layer_1_dropout']))

        # Layer Two 
        self.model.add(Conv2D(self.param['layer_2_neuron'], (3, 3), activation = self.param['layer_2_activation']))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(self.param['layer_2_dropout']))

        # Layer Three
        self.model.add(Conv2D(self.param['layer_3_neuron'], (3, 3), activation = self.param['layer_3_activation']))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(self.param['layer_3_dropout']))

        # Layer Four
        self.model.add(Conv2D(self.param['layer_4_neuron'], (3, 3), activation = self.param['layer_4_activation']))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(self.param['layer_4_dropout']))

        # Layer Five
        self.model.add(Flatten())
        self.model.add(Dense(self.param['layer_5_dense'], activation = self.param['layer_5_activation']))
        self.model.add(Dropout(self.param['layer_5_dropout']))

        # Final Layer
        self.model.add(Dense(self.outputData['Label'].shape[1], activation='softmax'))

        # Compile and fit
        self.model.compile(loss = 'categorical_crossentropy', optimizer = self.param['optimizer'])
        self.model.fit_generator(XYdataGenerator(self.inputData['image'],self.outputData['Label'],64,64,self.param['batch_size']),
                                 steps_per_epoch=int(ceil((len(self.inputData['image'])/self.param['batch_size']))),
                                 epochs=self.param['epochs'])

    def predictAlgo(self):
        r = self.model.predict_generator( XdataGenerator(self.inputData['image'], 64, 64, self.param['batch_size']),
                                          steps=int(ceil((len(self.inputData['image'])/self.param['batch_size']))))
        self.result['Label'] = r