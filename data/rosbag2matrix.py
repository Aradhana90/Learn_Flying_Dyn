import os.path
import pandas as pd
import seaborn as sea
from bagpy import bagreader

object = 'benchmark_box'
topic_name = 'Benchmark_box_DARKO'
run = 'small_dist'

path = 'raw/' + object + '/' + run

# Check if corresponding folder exists in extracted
if not os.path.isdir('extracted/' + object + '/' + run):
    os.mkdir('extracted/' + object + '/' + run)

# For every rosbag file in folder
idx = 0
for filename in os.listdir(path):
    if filename.endswith('.bag'):
        b = bagreader(path + '/' + filename)
        for t in b.topics:
            data = b.message_by_topic('/vrpn_client_node/' + topic_name)
        idx = idx+1
        print(idx)
