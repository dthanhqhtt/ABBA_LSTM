import unittest
from forecaster import forecaster
from batchless_VanillaLSTM_pytorch import batchless_VanillaLSTM_pytorch
from batchless_VanillaLSTM_keras import batchless_VanillaLSTM_keras
from VanillaLSTM_keras import VanillaLSTM_keras
from ABBA import ABBA as ABBA
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error as mse

start_time = time.time()

# Load data
df = pd.read_csv('/Users/prochang/Downloads/DataWater_train.csv', error_bad_lines=False)

# Fill the missing values with the mean values of each column
groups=[3,4,5,6,7,8,9,10,11,12]
for group in groups:
        mean_value = df.iloc[:, group].mean()
        df.iloc[:, group].fillna(value=mean_value, inplace=True)


#Choose the column data to train

i = 3  
# 3   pH 
# 4   EC        
# 5   DO        
# 6   TSS       
# 7   TN        
# 8   TP        
# 9   TOC       
# 10  ORP       
# 11  Temp      
# 12  TEMP      

data = df.iloc[:,i].to_list()


# k is number of step to forecast
k = 100
train_data = data[:-100]
test_data = data[-100:]


# Train
time_series = train_data
f = forecaster(time_series, model=VanillaLSTM_keras(), abba=ABBA(max_len=1000, verbose=2))
# can use other model like batchless_VanillaLSTM_keras or 
# batchless_VanillaLSTM_pytorch with 'stateful' & 'stateless' training.
# example : model = batchless_VanillaLSTM_pytorch(statefull = True)
f.train(max_epoch=100, patience=50)
prediction = f.forecast(k).tolist()


#Calculate running time
end_time = time.time()
elapsed_time = end_time - start_time
print("Time elapsed: ", elapsed_time, " seconds")

#Print Performance
err = np.sqrt(mse(prediction,test_data))
print('rmse = ', err)
