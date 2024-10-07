import os
from obspy import read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv

import os
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import os
import shutil
from obspy import read

class MSEEDLoader:
    def __init__(self, mseed_file):
        self.mseed_file = mseed_file
        self.data, self.times, self.sampling_rate, self.starttime = self.load_mseed_data()

    def load_mseed_data(self):
        """
        Load and preprocess data from a .mseed file.
        """
        st = read(self.mseed_file)
        tr = st[0]  
        data = tr.data
        times = tr.times()
        sampling_rate = tr.stats.sampling_rate
        starttime = tr.stats.starttime.datetime

        data = (data - np.mean(data)) / np.std(data)
        
        return data, times, sampling_rate, starttime

class TrainingDataPreparer:
    def __init__(self, mseed_directory, label_file):
        self.mseed_directory = mseed_directory
        self.label_file = label_file
        self.X = []
        self.y = []
        self.half_size = None

    def prepare_data(self):
        labels = pd.read_csv(self.label_file)
        min_size = min([len(MSEEDLoader(os.path.join(self.mseed_directory, row['filename'] + '.mseed')).data) for _, row in labels.iterrows()])

        self.half_size = min_size // 2

        for _, row in labels.iterrows():
            mseed_file = os.path.join(self.mseed_directory, row['filename'] + '.mseed')
            loader = MSEEDLoader(mseed_file)
            arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], '%Y-%m-%dT%H:%M:%S.%f')
            arrival = (arrival_time - loader.starttime).total_seconds()

            data = loader.data
            half = self.half_size


            quake_window, no_window = (data[0:half],data[half:half*2])  if (arrival in range(0,half)) else (data[half:half*2],data[0:half])

            self.X.append(quake_window)
            self.X.append(no_window)
            self.y.append(1)
            self.y.append(0)


        print(self.half_size)
        return np.array(self.X), np.array(self.y), self.half_size

class QuakeModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.half_size = None

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, stratify=y)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))
        self.plot_confusion_matrix(y_test, y_pred)
        return self.model

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        classes = unique_labels(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="viridis", ax=ax)
        ax.set_xticklabels(classes, rotation=30, ha='right')
        ax.set_yticklabels(classes, rotation=0)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        plt.show()

    def predict(self, feature_vector):
        return self.model.predict(feature_vector)


class DataPreprocessor:
    def __init__(self, half_size):
        self.half_size = half_size

    def split_list(self, input_list):
        return np.array([input_list[i:i + self.half_size] for i in range(0, len(input_list), self.half_size) if len(input_list[i:i + self.half_size]) == self.half_size])

    def preprocess_new_data(self, mseed_file):
        loader = MSEEDLoader(mseed_file)
        feature_vector = loader.data
        feature_vector = self.split_list(feature_vector)
        return feature_vector


class QuakeDetector:
    def __init__(self, model, half_size, output_directory):
        self.model = model
        self.data_preprocessor = DataPreprocessor(half_size)
        self.output_directory = output_directory

        for file_name in os.listdir(output_directory):
            if file_name.endswith('.mseed'):
                file_path = os.path.join(output_directory, file_name)
                os.remove(file_path)
        
        for file_name in os.listdir('./test_data_with_detected_quakes_graphs/'):
            file_path = os.path.join('./test_data_with_detected_quakes_graphs/', file_name)
            os.remove(file_path)


    def detect_quakes(self, test_data_directory):
        catalog_file = os.path.join(self.output_directory, 'quake_catalog.csv')
        
        with open(catalog_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'time_abs(%Y-%m-%dT%H:%M:%S.%f)', 'time_rel(sec)', 'evid', 'mq_type'])

            evid_counter = 1  

            for filename in os.listdir(test_data_directory):
                new_data_dir = f"{test_data_directory}{filename}"
                for mseed_file in os.listdir(new_data_dir):
                    if mseed_file.endswith('.mseed'):
                        file_path = os.path.join(new_data_dir, mseed_file)
                        X_newtest = self.data_preprocessor.preprocess_new_data(file_path)

                        loader = MSEEDLoader(file_path)
                        start_time = loader.starttime  

                        predictions = self.model.predict(X_newtest)

                        for i, prediction in enumerate(predictions):
                            if prediction == 1:
                                shutil.copy(file_path, self.output_directory)
                                print(f"Quake detected in interval {i} of: {mseed_file}")
                                
                                half_window_duration = self.data_preprocessor.half_size / loader.sampling_rate  
                                relative_time = i * half_window_duration  
                                absolute_time = start_time + timedelta(seconds=relative_time)  
                                evid = f"evid{evid_counter:05d}"
                                
                                writer.writerow([mseed_file, absolute_time.strftime('%Y-%m-%dT%H:%M:%S.%f'), int(relative_time), evid, 'impact_mq'])

                                evid_counter += 1


    def filter_detected_files(self):
        for file_name in os.listdir(self.output_directory):
            if file_name.endswith('.mseed'):
                file_path = os.path.join(self.output_directory, file_name)
                st = read(file_path)

                st_filt = st.copy()
                st_filt.filter('bandpass', freqmin=0.6, freqmax=0.9)
                tr_filt = st_filt[0]
                tr_times_filt = tr_filt.times()
                tr_data_filt = tr_filt.data

                fig, ax = plt.subplots(1, 1, figsize=(10, 3))
                ax.plot(tr_times_filt, tr_data_filt)
                ax.set_xlim([min(tr_times_filt), max(tr_times_filt)])
                ax.set_ylabel('Velocity (m/s)')
                ax.set_xlabel('Time (s)')
                ax.set_title(f'{file_name}', fontweight='bold')
                fig.savefig(f"test_data_with_detected_quakes_graphs/{file_name}_graph.png")
                plt.close(fig)

if __name__ == "__main__":
    mseed_directory = "./data/lunar/training/data/S12_GradeA/"
    label_file = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
    preparer = TrainingDataPreparer(mseed_directory, label_file)
    X, y, half_size = preparer.prepare_data()

    quake_model = QuakeModel()
    quake_model.train(X, y)

    quake_detector = QuakeDetector(quake_model, half_size, './results/')
    quake_detector.detect_quakes('./data/lunar/test/data/')
    quake_detector.filter_detected_files()
