import numpy as np 
import pandas as pd 
from scipy import signal
import matplotlib.pyplot as plt 
import math
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class Dataset:
    def __init__(self, data_file):
        self.data_file = data_file
        self.file_address = pd.read_csv('dataset/'+str(data_file), sep=',', header=None)

class Action(Dataset):
    def __init__(self, action, data_file):
        super().__init__(data_file)
        self.action_dictionary = {'sitting' : 1, 'lying' : 2,
                                  'standing' : 3, 'washing dishes' : 4,
                                  'vacuuming' : 5, 'sweeping' : 6,
                                  'walking outside' : 7, 'ascending stairs' : 8,
                                  'descending stairs' : 9, 'treadmill running' : 10,
                                  'bicycling (50 watt)' : 11, 'bicycling (100 watt)' : 12,
                                  'rope jumping' : 13}
        try:
            if(action in self.action_dictionary):
                self.action = str(action)
                self.action_val = self.file_address[self.file_address[24] == self.action_dictionary[self.action]].values
            else:
                raise IOError;
        except IOError:
            print(action+': Not a valid action.')

    def get_action(self):
        try:
            if(self.action != None):
                return self.file_address[self.file_address[24] == self.action_dictionary[self.action]];
            else:
                raise IOError;
        except IOError:
            print('An action does not exist.')

class Sensor(Action):
    
    def __init__(self, sensor, action, data_file):
        super().__init__(action, data_file)
        self.sensor_dictionary = {'wrist': [0,1,2,3,4,5],
                                  'chest' : [6, 7, 8, 9, 10, 11],
                                  'hip' : [12, 13, 14, 15, 16, 17],
                                  'ankle' : [18, 19, 20, 21, 22, 23]}
        try:
            if(sensor in self.sensor_dictionary):
                self.sensor = str(sensor)
                self.sensor_val = self.get_action().values[:, self.sensor_dictionary[self.sensor]]
            else:
                raise IOError;
        except IOError:
            print(sensor+': Not a valid sensor.')

    def get_sensor(self):
        try:
            if(self.sensor != None and self.action != None):
                return self.get_action()[:, self.sensor_dictionary[self.sensor]];
            else:
                raise IOError;
        except IOError:
            print('A sensor does not exist.')

    def data_visualization(self):
        plt.plot(self.action_val[:, self.sensor_dictionary[self.sensor][:3]])
        plt.title('Acceleration: x, y, z for ' + self.sensor)
        plt.show()

        plt.plot(self.action_val[:, self.sensor_dictionary[self.sensor][3:]])
        plt.title('Rotation: x, y, z for ' + self.sensor)
        plt.show()

    def noise_removing(self, filter_type, low=None, high=None):
        new_sensor = Sensor(self.sensor, self.action, self.data_file)
        if(filter_type == 'highpass' or filter_type == 'lowpass'):
            b, a = signal.butter(4, 0.04, filter_type, analog=False)
        elif(filter_type == 'bandpass' or filter_type == 'bandstop'):
            b, a = signal.butter(4, [low, high], filter_type, analog=False)
        for i in new_sensor.sensor_dictionary[new_sensor.sensor]:
            new_sensor.action_val[:, i] = signal.lfilter(b, a, new_sensor.action_val[:, i])
        new_sensor.data_visualization()
