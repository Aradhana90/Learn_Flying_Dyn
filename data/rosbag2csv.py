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

which_object = 'white_box'
topic_name = 'White_Box_DARKO/pose'
run = 'small_dist'

path = 'raw/' + which_object + '/' + run

# Check if corresponding folder exists in extracted
if not os.path.isdir('extracted/' + which_object + '/' + run):
    os.mkdir('extracted/' + which_object + '/' + run)

if __name__ == "__main__":
    # For every rosbag file in folder
    idx = 1
    csvfile = []
    for filename in os.listdir(path):
        if filename.endswith('.bag'):
            b = bagreader(path + '/' + filename)
            data = b.message_by_topic('/vrpn_client_node/' + topic_name)
            shutil.copyfile(data, 'extracted/' + which_object + '/' + run + '/' + str(idx) + '.csv')
            # csvfile.append(data)
            idx = idx + 1
