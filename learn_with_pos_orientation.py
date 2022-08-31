import pandas as pd
import csv, os.path
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
from sklearn.svm import SVR
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from sklearn.gaussian_process import GaussianProcessRegressor


def getDF(filename):
    df_main = pd.DataFrame(columns=['dt','x','y','z','xdot','ydot','zdot','f1','f2','f3','a1', 'a2', 'a3','a1dot', 'a2dot', 'a3dot','a1ddot', 'a2ddot', 'a3ddot'])
    for i in range(0,len(filename)):
        datafile = os.path.join(os.getcwd(),filename[i], "vrpn_client_node-darkoBox-pose.csv")
        df = pd.read_csv(datafile)
        list=['Time','pose.position.x', 'pose.position.y', 'pose.position.z' ]
        data=np.array([df[list[1]] ,df[list[2]],df[list[3]]])
        timestamps= np.array([df[list[0]]])
        # print('shape of data initially',data.shape)
        # initpt=data[:,0]
        # initpt = initpt.reshape(3,1)
      

        inipt= np.array([1.07855344, 1.44855011, 0.96767521]).reshape(3,1)
        data= data-inipt
        # print('shape of modified data',data.shape)
        diffdata, deltat=np.diff(data), np.diff(timestamps)
        data_vel= diffdata/deltat

        data_vel = savgol_filter(data_vel, 21, 5)
        # print(data_vel)
        data_acc= np.diff(data_vel)/deltat[0,:-1]
        # plt.plot(data_acc[0,:])
        data_acc = savgol_filter(data_acc, 21, 5)
        # plt.plot(data_acc[0,:])
        # shiftdiffdata= np.roll(diffdata,1)
        iparraysize=len(diffdata[0,:])-1
        iparray=np.zeros(iparraysize)
        # print('diffarray ip is ',diffdata)
        for i in range(0, len(diffdata[1,:])-1):
            dotprod=np.dot(diffdata[:,i],diffdata[:,i+1])
            norms= np.linalg.norm(diffdata[:,i])*np.linalg.norm(diffdata[:,i+1])
            # print( dotprod/ (norms))
            iparray[i]= dotprod/norms
        # iparray= np.diag(np.matmul(np.transpose(diffdata), shiftdiffdata))
        # iparray= np.einsum('ii->i', np.einsum('ij,jk',np.einsum('ji', shiftdiffdata),diffdata ))
        # print('iparray is ',iparray[400:431])
        # signiparray= np.sign(iparray)
        # print('sign ip is ',signiparray)

        ### smoothness check
        startat=0
        breakat= len(iparray)-5
        for i in range(0, len(iparray)-1): 
            # print(val.shape)
            if  i+5<len(iparray):
                val =iparray[i:i+5]
                if (val-0.98*np.ones(5) >0).all() and startat==0:
                    startat= i
            if  iparray[i]<0.89 and startat!=0:
                breakat=i
                break

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1,projection='3d')
        # # print(saved_column)
        # # indexes= range(1, len(saved_column)+1)
        # # ax.scatter3D(data[0,0:40], data[1,0:40], data[2,0:40])
        # # ax = fig.add_subplot(1, 3, 2,projection='3d')
        # # ax.scatter3D(data[0,0:50], data[1,0:50], data[2,0:50])
        # # ax = fig.add_subplot(1, 3, 3,projection='3d')
        # # ax.scatter3D(data[0,0:100], data[1,0:100], data[2,0:100])
        # # ax.scatter3D(diffdata[0,:], diffdata[1,:], diffdata[2,:])
        # # ax.scatter3D(data[0,:], data[1,:], data[2,:])
        # ax.scatter3D(data[0,startat:breakat], data[1,startat:breakat], data[2,startat:breakat])
        # Mat=[data[0,startat:breakat], data[1,startat:breakat], data[2,startat:breakat],
        #     diffdata[0,startat:breakat], diffdata[1,startat:breakat],diffdata[2,startat:breakat]]
        # data_main.append(Mat)
        df_quat = df[['pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z','pose.orientation.w' ]].copy()
        rot = Rotation.from_quat(df_quat)
        rot_euler = rot.as_euler('xyz', degrees=True)
        euler_df = pd.DataFrame(data=rot_euler, columns=['a1', 'a2', 'a3'])
        diff_euler_df= euler_df.diff()
        ddiff_euler_df= diff_euler_df.diff()
        
        
        euler_df = euler_df.iloc[startat:breakat]

        diff_euler_df= diff_euler_df.iloc[startat:breakat]
        diff_euler_df= diff_euler_df.rename({'a1': 'a1dot', 'a2':'a2dot', 'a3':'a3dot'}, axis='columns')
        
        
        ddiff_euler_df= ddiff_euler_df.iloc[startat:breakat]
        ddiff_euler_df= ddiff_euler_df.rename({'a1': 'a1ddot', 'a2':'a2ddot', 'a3':'a3ddot'}, axis='columns')
        
        
        Euler_df = pd.concat([euler_df, diff_euler_df,ddiff_euler_df ], axis=1)
        Euler_df = Euler_df.reset_index(drop=True)
        

        df = pd.DataFrame({'dt':deltat[0,startat:breakat] ,'x': data[0,startat:breakat], 'y':  data[1,startat:breakat], 'z':  data[2,startat:breakat],
        'xdot': data_vel[0,startat:breakat], 'ydot':  data_vel[1,startat:breakat], 'zdot':  data_vel[2,startat:breakat],
        'f1': data_acc[0,startat:breakat], 'f2':  data_acc[1,startat:breakat], 'f3':  data_acc[2,startat:breakat]})
        
        df = pd.concat([df, Euler_df], axis=1)
      

        df_main= pd.concat([df_main, df])
        # print(df_main.keys())
    
    # df_main=  pd.DataFrame(data_main)
    # ax.quiver(data[0,:-1], data[1,:-1], data[2,:-1], diffdata[0,:],diffdata[1,:], diffdata[2,:], length=1)
    # plt.show()
    X_nxt=[]
    
    X = df_main[['x','y','z','a1', 'a2', 'a3']].to_numpy()
    Xdot= df_main[['xdot','ydot','zdot','a1dot', 'a2dot', 'a3dot']].to_numpy()
    Xddot= df_main[['f1','f2','f3','a1ddot', 'a2ddot', 'a3ddot' ]].to_numpy()
    deltat= df_main[['dt']].to_numpy()
    
    

    Xinit= np.append(X[0,:],Xdot[0,:]).reshape(1,12)
    X_plt=Xinit
    # print(np.size(deltat))
    for i in range(0, np.size(deltat)-1):
        del_t = deltat[i,0]
        # print(X_plt.shape)
        nxt_vel= X_plt[-1,6:12]+Xddot[i,:] *del_t
        nxt_pos= X_plt[-1,0:6]+ X_plt[-1,6:12]*del_t
        X_nxt = np.append(nxt_pos, nxt_vel).reshape(1,12)
        X_plt = np.append( X_plt, X_nxt , axis=0)
        

    df_main.to_csv('outfile.csv')
    return [df_main, X_plt]



def get_training_data(df_main):
    X= df_main[['x','y','z','a1', 'a2', 'a3','xdot', 'ydot','zdot','a1dot', 'a2dot', 'a3dot']].to_numpy()
    Y1, Y2, Y3= df_main[['f1']].to_numpy(), df_main[['f2']].to_numpy() , df_main[['f3']].to_numpy() 
    Y4, Y5, Y6 = df_main[['a1ddot']].to_numpy(), df_main[['a2ddot']].to_numpy() , df_main[['a3ddot']].to_numpy()
    # training_indices = np.random.randint(0, np.size(Y1,0)-1, size=300)
    training_indices= range(0, np.size(Y1,0))
    # print(training_indices)


    # X_train, y_train = X[training_indices], Y[training_indices]
    # X_train , y1_train, y2_train, y3_train= X[range(0, 53)], Y1[range(0,53)], Y2[range(0,53)],Y3[range(0,53)]
    X_train = X[training_indices,:]
    # print(X_train)
   
    y1_train, y2_train, y3_train= Y1[training_indices], Y2[training_indices],Y3[training_indices]
    y4_train, y5_train, y6_train= Y4[training_indices], Y5[training_indices],Y6[training_indices]
    
    # scaler = preprocessing.StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # plt.plot(Y1)
    return [X_train , y1_train, y2_train, y3_train,y4_train, y5_train, y6_train]



def get_GP(X_train, y1_train, y2_train, y3_train,y4_train, y5_train, y6_train):
    kernel = 2* RBF(length_scale=2.0, length_scale_bounds=(1e-3, 1e3))
    kernel3 = 1.5* RBF(length_scale=2.5, length_scale_bounds=(1e-3, 1e3))
    GP1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    GP2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    GP3 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    GP4 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    GP5 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    GP6 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    GP1.fit(X_train, y1_train)
    GP2.fit(X_train, y2_train)
    GP3.fit(X_train, y3_train)
    GP4.fit(X_train, y4_train)
    GP5.fit(X_train, y5_train)
    GP6.fit(X_train, y6_train)
    # print(gaussian_process.kernel_)
    # print(training_indices.size)
    # remaining_indices=[]
    # for i in range(0,np.size(Y,0)-1):
    #     c=0
    #     for j in range(0,training_indices.size-1):
    #         if training_indices[j] ==i:
    #             c=1
    #     if c==0:
    #         remaining_indices.append(i)
    return [GP1, GP2, GP3, GP4, GP5, GP6]


def get_SVR(X_train, y1_train, y2_train, y3_train, y4_train, y5_train, y6_train, kerneltype):

    # SVR1 = SVR(kernel="poly", C=100, gamma="auto", degree=4, epsilon=0.1, coef0=1)
    # SVR2= SVR(kernel="poly", C=100, gamma="auto", degree=4, epsilon=0.1, coef0=1)
    # SVR3= SVR(kernel="poly", C=100, gamma="auto", degree=4, epsilon=0.1, coef0=1)
    SVR1 = SVR(kernel=kerneltype[0], C=100, gamma=0.5, epsilon=0.1)
    SVR2 = SVR(kernel=kerneltype[0], C=100, gamma=0.1, epsilon=0.1)
    SVR3 = SVR(kernel=kerneltype[0], C=100, gamma=0.5, epsilon=0.2)
    SVR4 = SVR(kernel=kerneltype[0], C=100, gamma=0.5, epsilon=0.2)
    SVR5 = SVR(kernel=kerneltype[0], C=100, gamma=0.5, epsilon=0.2)
    SVR6 = SVR(kernel=kerneltype[0], C=100, gamma=0.5, epsilon=0.2)
    # print(SVR1.__dir__.keys())
    SVR1.fit(X_train, y1_train)
    SVR2.fit(X_train, y2_train)
    SVR3.fit(X_train, y3_train)
    SVR4.fit(X_train, y4_train)
    SVR5.fit(X_train, y5_train)
    SVR6.fit(X_train, y6_train)
    # print(gaussian_process.kernel_)
    # print(training_indices.size)
    # remaining_indices=[]
    # for i in range(0,np.size(Y,0)-1):
    #     c=0
    #     for j in range(0,training_indices.size-1):
    #         if training_indices[j] ==i:
    #             c=1
    #     if c==0:
    #         remaining_indices.append(i)
    return [SVR1, SVR2, SVR3, SVR4, SVR5, SVR6]



def test(Mod1, Mod2, Mod3, Mod4, Mod5, Mod6, init_idx,filename):
    # print(remaining_indices)
    # data_remaining = df_main.iloc[remaining_indices]
    # X_test= df_main[['x','y','z']].to_numpy()
    # print(X_test)
    # X_test = df_main.iloc[range(0, 53)][['x','y','z']].to_numpy()
    # X_test = X[range(0, 53)]
    # Xinit= X[100]
    # mean_prediction= gaussian_process.predict(X_test, return_std=False)

    [df_main, X_plt]= getDF(filename)
    Xdata= df_main[['x','y','z','a1', 'a2', 'a3','xdot', 'ydot','zdot','a1dot', 'a2dot', 'a3dot']].to_numpy()
    # print(Xdata.shape)
    deltat=df_main[['dt']].to_numpy()
    Xinit= Xdata[init_idx,:]
    X_plt=np.array([Xinit])
    # print(np.squeeze(X_plt[-1,:], axis=0))
    X_nxt=[]
    acc_array=np.empty(shape=[1, 6])
    # print(np.size(deltat))
    for i in range(0, np.size(deltat)-1):
        # print(GP1.predict(X_plt[-1],return_std=False))
        # del_t=0.2
        del_t = deltat[i,0]
        
        # print(X_plt.shape)
        # nxt_vel= X_plt[-1,6:12]+Xddot[i,:] *del_t
        # nxt_pos= X_plt[-1,0:6]+ X_plt[-1,6:12]*del_t
        # X_nxt = np.append(nxt_pos, nxt_vel).reshape(1,12)
        # X_plt = np.append( X_plt, X_nxt , axis=0)
        

        acc= np.array([Mod1.predict(np.array([X_plt[-1]])), Mod2.predict(np.array([X_plt[-1]])), Mod3.predict(np.array([X_plt[-1]])),
        Mod4.predict(np.array([X_plt[-1]])), Mod5.predict(np.array([X_plt[-1]])), Mod6.predict(np.array([X_plt[-1]])) ])
        acc= acc.reshape(1,6)
        acc_array= np.append(acc_array, acc, axis=0)
        
        nxt_vel= np.squeeze(X_plt[-1,6:12])+np.squeeze(acc) *del_t
        nxt_pos= np.squeeze(X_plt[-1,0:6])+ np.squeeze(X_plt[-1,6:12])*del_t
        X_nxt = np.append(nxt_pos, nxt_vel)
        X_plt = np.append( X_plt, np.array([X_nxt]) , axis=0)
        # print(X_plt.shape)
    # ax = plt.subplot(projection='3d')
    # ax = Axes3D(fig)
    # print(X_test.shape)
    # plt.scatter(range(0,3),range(0,3),range(0,3))
    # ax.scatter3D(X_test[100:, 0], X_test[100:, 1], X_test[:, 2])
    # ax.scatter3D(X_train[init_idx:10, 0], X_train[init_idx:10, 1], X_train[init_idx:10, 2])
    # print(acc_array)
    return(X_plt, acc_array)

def plot(total_num,num, X_train, X_plt):
    ax = fig.add_subplot(1, total_num, num, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter3D(X_train[:, 0], X_train[:, 1], X_train[:, 2])
    ax.scatter3D(X_plt[:, 0], X_plt[:, 1], X_plt[:, 2])


fig = plt.figure()
# plt.legend()
# filename=['2022-07-06-15-46-37', '2022-07-06-15-46-57', '2022-07-06-15-51-00'
# , '2022-07-06-15-53-09', '2022-07-06-15-55-10','2022-07-06-15-55-57','2022-07-06-15-56-30']

### poor performance of both GP and SVR, z decreasing
### '2022-07-06-15-50-32', '2022-07-06-15-51-00','2022-07-06-15-51-35'


### z increasing, good performance of GP to some extent (time step issue? marker initial position upside down?)
### '2022-07-06-15-52-18', '2022-07-06-15-52-35','2022-07-06-15-52-52', '2022-07-06-15-54-49', '2022-07-06-15-55-10'


### z incresing filenames rejected
### '2022-07-06-15-51-55','2022-07-06-15-53-33' '2022-07-06-15-54-06','2022-07-06-15-54-24'

### too few points sampled 
### '2022-07-06-15-53-52'

filename=[ '2022-07-06-15-46-57']
timesteps, init_idx, del_t =30, 0, 0.3

[df_main, X_plt] = getDF(filename)
[X_train , y1_train, y2_train, y3_train, y4_train, y5_train, y6_train ] = get_training_data(df_main)

# plot(1,1, X_train,X_plt)


#### train with various methods
#### train , test with GP and plot
testfile=['2022-07-06-15-46-57']
[Mod1, Mod2, Mod3,Mod4, Mod5, Mod6]= get_GP(X_train, y1_train, y2_train, y3_train,y4_train, y5_train, y6_train)
[X_plt, acc_array] = test(Mod1, Mod2, Mod3, Mod4, Mod5, Mod6, init_idx,testfile)


# MSE = np.sum(np.square(X_plt[:,0:3]- X_train[:,0:3]), axis = 1)
# print('MSE is',MSE)
# plot(2,1, X_train, X_plt)


# plt.plot(Xddot[10:,1], acc_array[10:,1])



# train with SVR

kerneltype= ['poly']

#### test and plot
#### test with SVR and plot
# [Mod1, Mod2, Mod3] = get_SVR(X_train, y1_train, y2_train, y3_train,y4_train, y5_train, y6_train,kerneltype)
# X_plt = test(Mod1, Mod2, Mod3, Mod4, Mod5, Mod6, init_idx, testfile)
# plot(2,2, X_train, X_plt)

# Xddot= df_main[['f1','f2','f3','a1ddot', 'a2ddot', 'a3ddot' ]].to_numpy()
# # print(df_main.shape)
# print(Xddot[:,0])
# print('predict acc is', acc_array[:,0])


plt.show()
# data_main=[]

