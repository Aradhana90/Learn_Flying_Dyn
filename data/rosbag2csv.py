import csv
import os.path
import shutil
import sys

import bagpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sea
from bagpy import bagreader

which_object = 'benchmark_box'
topic_name = 'Benchmark_Box_DARKO/pose'
run = 'small_dist'

path = 'raw/' + which_object + '/' + run

# Check if corresponding folder exists in extracted
if not os.path.isdir('extracted/' + which_object + '/' + run):
    os.mkdir('extracted/' + which_object + '/' + run)

# For every rosbag file in folder
idx = 1
csvfile = []
for filename in os.listdir(path):
    if filename.endswith('.bag'):
        b = bagreader(path + '/' + filename)
        data = b.message_by_topic('/vrpn_client_node/' + topic_name)
        shutil.copyfile(data, 'extracted/' + which_object + '/' + run + '/' + str(idx) + '.csv')
        # csvfile.append(data)
        idx = idx+1

# for t in b.topics:
#     data = b.message_by_topic(t)
#     csvfiles.append(data)
#     # # Objectname: darkoBox
# datafile = os.path.join(os.getcwd(),"2022-08-26-15-18-39", "vrpn_client_node-darkoBox-pose.csv")
# df = pd.read_csv(datafile)
# print (df.columns)
# list=['Time','pose.position.x', 'pose.position.y', 'pose.position.z', 'pose.orientation.x', 'pose.orientation.y',
#        'pose.orientation.z', 'pose.orientation.w']
# # pos=np.zeros((1,len(df['pose.position.x'])) )
# # saved_column = df['pose.position.x']

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # print(saved_column)
# # indexes= range(1, len(saved_column)+1)
# plt.scatter(df[list[1]], df[list[2]],df[list[3]])

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()