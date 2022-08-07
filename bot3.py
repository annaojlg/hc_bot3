# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 11:22:45 2021

@author: moham
"""
import argparse
import sys
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import importlib
import tensorflow as tf

if __name__ == "__main__":
    myargparser = argparse.ArgumentParser()
    myargparser.add_argument('--maxtime', type=int, const=120, nargs='?', default=120)
    myargparser.add_argument('--mode', type=int, const=120, nargs='?', default=1) # 1: Stress Detection, 2: Action evaluation
    myargparser.add_argument('--bot_id', type=str, const='text', nargs='?', default=1)
    myargparser.add_argument('--data_api', type=str, const='text', nargs='?', default='/robothon/mk7744/healthcare_roboplatform/bots/')
    myargparser.add_argument('--event_id', type=int, const=1, nargs='?', default=1)
    myargparser.add_argument('--result_path', type=str, const='text', nargs='?', 
                             default='D:/OneDrive - Higher Education Commission/Documents/NYU/Semester 3/ITP/Bots repos/models/bot')
    
    args = myargparser.parse_args()
    sys.path.append(args.data_api)
    bot_data = importlib.import_module('bot_data')

    print(args.maxtime)
    print(args.bot_id)
    print(args.data_api)
    if args.mode==1:
        mybotdata = bot_data.BotData()
        result = mybotdata.fetchdataframe(1)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        dataset=result.iloc[:,2:]
        dataset['hour']=dataset._time.dt.hour
        dataset['mins']=dataset._time.dt.minute
        X = dataset[['calories','distance','heart_rate','steps','hour','mins']].values
        y = dataset[['stress']].values
        print(X.shape,y.shape)
        X_train,y_train = X,y
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        #X_test = sc.transform(X_test)
        classifier = Sequential()
        classifier.add(Dense(10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
        classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
        filelocation = args.result_path
        classifier.save(filelocation)
    elif args.mode==2:
        mybotdata = bot_data.BotData()
        result = mybotdata.fetchdataframe(1)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        dataset=result.iloc[:,2:]
        dataset['hour']=dataset._time.dt.hour
        dataset['mins']=dataset._time.dt.minute
        dataset['next_stress'] = dataset['stress'].shift(-1)
        #dataset.loc[dataset.stress==1 and dataset.next_stress==0,'stress_red'] = 1
        #dataset.loc[dataset.stress==1 and dataset.next_stress==1,'stress_red'] = 0
        #dataset[dataset.stress==1 and (dataset.stress_red==0 or dataset.stress_red==1)]
        dataset = dataset.loc[(dataset['stress']==1) & (dataset['next_stress']==0)]
        dataset = dataset.dropna()
        X = dataset[['calories','distance','heart_rate','steps','hour','mins']].values
        y = dataset[['activity']].values
        sc = StandardScaler()
        X = sc.fit_transform(X)
        classifier = Sequential()
        classifier.add(Dense(30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
        classifier.add(Dense(100, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(10))
        classifier.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
        classifier.fit(X, y, batch_size = 32, epochs = 100)
        filelocation = args.result_path
        classifier.save(filelocation)
        print(X.shape,y.shape)
        
        #X = dataset[['calories','distance','heart_rate','steps','hour','mins']].values
        #y = dataset[['stress']].values