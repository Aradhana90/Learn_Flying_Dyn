import csv
import os.path
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
csvfile = []
for filename in os.listdir(path):
    if filename.endswith('.bag'):
        b = bagreader(path + '/' + filename)
        data_csv = b.message_by_topic('/vrpn_client_node/' + topic_name)
        df = pd.read_csv(data_csv)
        tmp = ['Time', 'pose.position.x', 'pose.position.y', 'pose.position.z', 'pose.orientation.x']
        data_pos = np.array([df[tmp[1]], df[tmp[2]], df[tmp[3]]])
        print('bla')

