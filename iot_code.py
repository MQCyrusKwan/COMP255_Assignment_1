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

#Author: Cyrus Kwan    45200165
#        cyrus.kwan@students.mq.edu.au
#Date: 15 September 2019

class Dataset:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data_set = pd.read_csv('dataset/'+str(data_file), sep=',', header=None)
        self.data_frame = self.data_set.loc
        self.data_val = self.data_set.values

    def noise_removing(self, filter_type, range_from, range_to=None):
        '''applies filter to parsed class'''
        if(filter_type == 'highpass' or filter_type == 'lowpass'):
            b, a = signal.butter(4, range_from, filter_type, analog=False)
        elif(filter_type == 'bandpass' or filter_type == 'bandstop'):
            b, a = signal.butter(4, [range_from, range_to], filter_type, analog=False)

        filter_val = self.data_val
        if(isinstance(self, Dataset)):
            if(isinstance(self, Action)):
                filter_val = self.action_val
                if(isinstance(self, Sensor)):
                    filter_val = self.sensor_val
                    for sensor in dictionary('sensor')[self.sensor]:
                        filter_val[:, sensor] = signal.lfilter(b, a, filter_val[:, sensor])
                    return filter_val;
                for action in dictionary('action'):
                    filter_val[:, dictionary('action')[action]] = signal.lfilter(b, a, filter_val[:, dictionary('action')[action]])
                return self.filter_val();
            for columns in range(filter_val.shape[-1]):
                filter_val[:, columns] = signal.lfilter(b, a, filter_val[:, columns])
            return filter_val;

class Action(Dataset):
    def __init__(self, action, data_file):
        super().__init__(data_file)
        
        try:
            if(action in dictionary('action')):
                self.action = str(action)
                self.action_val = self.data_val[self.data_val[:, 24] == dictionary('action')[self.action]]
            else:
                raise IOError;
        except IOError:
            print(action+': Not a valid action.')

    def get_action(self):
        '''returns a static dataframe of the specific action attatched to the class'''
        try:
            if(self.action != None):
                return self.data_set[self.data_frame[:, 24] == dictionary('action')[self.action]];
            else:
                raise IOError;
        except IOError:
            print('An action does not exist.')

class Sensor(Action):
    def __init__(self, sensor, action, data_file):
        super().__init__(action, data_file)
        
        try:
            if(sensor in dictionary('sensor')):
                self.sensor = str(sensor)
                self.sensor_val = self.get_action().values[:, dictionary('sensor')[self.sensor]]
            else:
                raise IOError;
        except IOError:
            print(sensor+': Not a valid sensor.')

    def get_sensor(self):
        '''returns a static dataframe of the specific sensor attatched to the class'''
        try:
            if(self.sensor != None and self.action != None):
                return self.get_action()[:, dictionary('sensor')[self.sensor]];
            else:
                raise IOError;
        except IOError:
            print('A sensor does not exist.')

    def data_visualization(self):
        '''plots and displays graphs of acceleration and rotation of the repective sensor group'''
        plt.plot(self.action_val[:, dictionary('sensor')[self.sensor][:3]])
        plt.title('Acceleration: x, y, z for '+self.sensor+' while '+self.action)
        plt.show()

        plt.plot(self.action_val[:, dictionary('sensor')[self.sensor][3:]])
        plt.title('Rotation: x, y, z for '+self.sensor+' while '+self.action)
        plt.show()

def dictionary(dictionary_name):
    '''returns a dictionary cointaining either a sensor's respective columns or an action's
    index value'''
    try:
        if(dictionary_name == 'action'):
            action_dictionary = {'sitting' : 1, 'lying' : 2,
                                 'standing' : 3, 'washing dishes' : 4,
                                 'vacuuming' : 5, 'sweeping' : 6,
                                 'walking outside' : 7, 'ascending stairs' : 8,
                                 'descending stairs' : 9, 'treadmill running' : 10,
                                 'bicycling (50 watt)' : 11, 'bicycling (100 watt)' : 12,
                                 'rope jumping' : 13}
            return action_dictionary;
        elif(dictionary_name == 'sensor'):
            sensor_dictionary = {'wrist': [0,1,2,3,4,5],
                                 'chest' : [6, 7, 8, 9, 10, 11],
                                 'hip' : [12, 13, 14, 15, 16, 17],
                                 'ankle' : [18, 19, 20, 21, 22, 23]}
            return sensor_dictionary;
        else:
            raise IOError;
    except IOError:
        print(dictionary_name+' is not a valid dictionary')
        
    
def feature_engineering(filter_type, range_from, range_to=None):
    '''writes processed data to training and testing .csv files'''
    training = np.empty(shape=(0, 10))
    testing = np.empty(shape=(0, 10))
    for file_index in range(19):
        print('dataset '+str(file_index+1))
        feature_set = Dataset('dataset_'+str(file_index+1)+'.txt')
        feature_data = feature_set.noise_removing(filter_type, range_from, range_to)
        feature_set.data_set.describe()

        datat_len = len(feature_data)
        training_len = math.floor(datat_len * 0.8)
        training_data = feature_data[:training_len, :]
        testing_data = feature_data[training_len:, :]

        # data segementation: for time series data, we need to segment the whole time series, and then extract features from each period of time
        # to represent the raw data. In this example code, we define each period of time contains 1000 data points. Each period of time contains 
        # different data points. You may consider overlap segmentation, which means consecutive two segmentation share a part of data points, to 
        # get more feature samples.
        training_sample_number = training_len // 1000 + 1
        testing_sample_number = (datat_len - training_len) // 1000 + 1

        for s in range(training_sample_number):
            if s < training_sample_number - 1:
                sample_data = training_data[1000*s:1000*(s + 1), :]
            else:
                sample_data = training_data[1000*s:, :]
            # in this example code, only three accelerometer data in wrist sensor is used to extract three simple features: min, max, and mean value in
            # a period of time. Finally we get 9 features and 1 label to construct feature dataset. You may consider all sensors' data and extract more

            feature_sample = []
            for i in range(3):
                feature_sample.append(np.min(sample_data[:, i]))
                feature_sample.append(np.max(sample_data[:, i]))
                feature_sample.append(np.mean(sample_data[:, i]))
                feature_sample.append(sample_data[0, -1])
                feature_sample = np.array([feature_sample])
                training = np.concatenate((training, feature_sample), axis=0)
            
            for s in range(testing_sample_number):
                if s < training_sample_number - 1:
                    sample_data = testing_data[1000*s:1000*(s + 1), :]
                else:
                    sample_data = testing_data[1000*s:, :]

                feature_sample = []
                for i in range(3):
                    feature_sample.append(np.min(sample_data[:, i]))
                    feature_sample.append(np.max(sample_data[:, i]))
                    feature_sample.append(np.mean(sample_data[:, i]))
                feature_sample.append(sample_data[0, -1])
                feature_sample = np.array([feature_sample])
                testing = np.concatenate((testing, feature_sample), axis=0)

    df_training = pd.DataFrame(training)
    df_testing = pd.DataFrame(testing)
    df_training.to_csv('training_data.csv', index=None, header=None)
    df_testing.to_csv('testing_data.csv', index=None, header=None)
