import bagpy
from bagpy import bagreader
import sys

import csv, os.path
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np


b = bagreader('2022-08-26-15-18-39.bag')
csvfiles = []
for t in b.topics:
    data = b.message_by_topic(t)
    csvfiles.append(data)
datafile = os.path.join(os.getcwd(),"2022-08-26-15-18-39", "vrpn_client_node-darkoBox-pose.csv")
df = pd.read_csv(datafile)
print (df.columns)
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