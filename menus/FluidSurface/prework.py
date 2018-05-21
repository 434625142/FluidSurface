import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path as osp

def read_mat(path):
    data=sio.loadmat(path)
    for i in list(data.keys()):
        if '__' in i: del data[i]
    for i in data.keys():
        data[i] = data[i].ravel()
    Signal37 = data.pop('Signal37')
    data = pd.DataFrame(data)
    
    ns = np.arange(len(Signal37))*100
    data['Signal37'] = np.nan
    data['Signal37'][ns] = Signal37
    return data

def get_global_dt(data, img, n, senf=10000, imgf=25):
    imgloc = img['Y']
    senloc = data['Position']
    light = data['triggerlight']
    t = np.where(light>4)[0][0]
    t0 = int(t - n*senf/imgf)
    print('粗略开灯时间:%s'%t0)
    ds = (np.arange(len(imgloc))*(senf/imgf)).astype(np.int)

    maxcf, maxt, s = -10, t0, 0
    
    for i in range(-3000, 3000, 10):
        coef = np.corrcoef(-imgloc, senloc[ds+(t0+i)])[1,0]
        if coef > maxcf:
            maxcf, maxt = coef, t0+i
    
    print('精准开灯时间:%s'%maxt)
    senx = np.arange(len(senloc))
    seny = data['Position']
    seny -= seny.mean()
    seny /= seny.max()
    imgx = ds + maxt
    imgy = -imgloc
    imgy -= imgy.mean()
    imgy /= imgy.max()
    plt.plot(senx, seny)
    plt.plot(imgx, imgy)
    plt.show()

def do_add_global_t0():
    print('drag data file here:')
    path = input()
    data = pd.read_csv(path)
    print('t0?:')
    t0 = int(input())
    print('n?:')
    n = int(input())
    data['time'] = np.arange(len(data))*400+(t0+n*400)
    p,name = osp.split(path)
    data.to_csv(osp.join(p,'_'+name))

def do_get_video_t0():
    print('drag data file here:')
    path = input()
    data = read_mat(path)
    print('drag img loc here:')
    path2 = input()
    imgloc = pd.read_csv(path2)
    print('n?:')
    n = int(input())
    get_global_dt(data, imgloc, n)

def do_get_local_t0():
    print('drag data file here:')
    path = input()
    data = read_mat(path)
    trig = data['triggerHexapode']>4
    t0 = np.where(trig[400000:])[0][0]+400000
    print('起始时间:%s'%t0)
    input()

def do_add_local_t0():
    print('drag data file here:')
    path = input()
    data = pd.read_csv(path)
    print('t0?:')
    t0 = int(input())
    data['time'] = np.arange(len(data))*100+t0
    p,name = osp.split(path)
    data.to_csv(osp.join(p,'_'+name))
    
if __name__ == '__main__':
    
    print('what would you do?')
    funs = {'gvt0':do_get_video_t0,
            'agt0':do_add_global_t0,
            'glt0':do_get_local_t0,
            'alt0':do_add_local_t0}
    print('gvt0:do_get_video_t0\nagt0:do_add_global_t0\nglt0:do_get_local_t0\nalt0:do_add_local_t0')
    funs[input()]()
    
    #data = read_mat('D9ATMF140.mat')
