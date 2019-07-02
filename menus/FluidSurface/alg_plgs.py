from imagepy.core.engine import Filter, Simple
from imagepy.ipyalg import watershed
from imagepy import IPy
import os.path as osp
import numpy as np
import scipy.ndimage as ndimg
import cv2
import pandas as pd

def combine(img):
    h,w = img.shape
    l, r = img[:,:w//2], img[:,w//2:]
    return np.hstack((l.T[::-1,:], r.T[:,::-1]))

class Combine(Simple):
    title = 'Re Combine'
    note = ['8-bit']
    
    #process
    def run(self, ips, imgs, para = None):
        for i in range(len(imgs)):
            imgs[i] = combine(imgs[i])
            self.progress(i, len(imgs))
        ips.set_imgs(imgs)

class Dark(Filter):
    title = 'Dark Little'
    note = ['all', 'auto_msk', 'auto_snap']

    def run(self, ips, snap, img, para = None):
        np.multiply(snap, 0.95, out=img, casting='unsafe')
        img += 1

class DOG(Filter):
    title = 'Fast DOG'
    note = ['all', 'auto_msk', 'auto_snap', 'preview']

    #parameter
    para = {'sigma':0}
    view = [(float,'sigma',  (0,30), 1,  'sigma', 'pix')]

    #process
    def run(self, ips, snap, img, para = None):
        l = int(para['sigma']*3)*2+1
        cv2.GaussianBlur(snap, (l, l), para['sigma'], dst=img)
        msk = img<snap
        img-=snap
        img[msk] = 0

class Gradient(Filter):
    title = 'Gradient From Bottom'
    note = ['all', '2int', 'auto_msk', 'auto_snap']
    #process
    def run(self, ips, snap, img, para = None):
        img[:] =  np.clip(ndimg.sobel(snap, axis=0, output=img.dtype), 0, 1e4)

class Watershed(Filter):
    title = 'Watershed Surface'
    note = ['8-bit', 'auto_snap', 'not_channel', 'preview']

    #process
    def run(self, ips, snap, img, para = None):
        markers = img*0
        markers[[0,-1]] = [[1],[2]]
        mark = watershed(img, markers, line=True, conn=1)
        img[:] = (mark==0) * 255

class Predict(Filter):
    model = None
    title = 'Predict Surface'
    note = ['8-bit', 'auto_snap',  'preview']
    mode_list=['msk','line','line on ori']
    view = [(list,'mode', mode_list, str, 'mode',  '')]
    para = {'mode':mode_list[0]}

    def load(self, ips):
        if not Predict.model is None: return True
        from keras.models import load_model
        try:
            path = osp.join(osp.abspath(osp.dirname(__file__)), 'U-net.h5')
            Predict.model=load_model(path)
        except Exception as e:
            IPy.alert('Not Found Net')
            return False
        #一定要预测一次，否则后面会出错        
        Predict.model.predict(np.zeros((1, 224,224,1)))     
        return True 

    def run(self, ips, snap, img, para = None):
        shape_temp=snap.shape
        temp=cv2.resize(snap,(224,224)).reshape(1,224,224,1).astype('float32')/255.0
        pred=(Predict.model.predict(temp)*255).astype('uint8').reshape(224,224)
        img[:]=(cv2.resize(pred,(shape_temp[1],shape_temp[0]))>127)*255
        
        if para['mode']=='msk':return
        line = cv2.dilate(img, np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))
        if para['mode']=='line':img[:] = line -img
        if para['mode']=='line on ori': np.max([snap, line-img], axis=0, out=img)

class Eliminate(Simple):
    modelist=['SQDIFF','SQDIFF_NORMED','CCORR','CCORR_NORMED','CCOEFF']
    title = 'Eliminate vibration'
    note = ['8-bit', 'req_roi']
    para = {'amp_x':2,
            'amp_y':60,
            'mode':modelist[3]
            }
    view = [(int, 'amp_x',(0,10), 0, 'amp_x',  ''),
            (int, 'amp_y',(0,80), 0, 'amp_y',  ''),
            (list, 'mode',modelist, str, 'mode',  '')
            ]
    #process
    def mathc_img(self,image,template,mode):
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(image,template,mode)
        loc = np.where( res ==res.max())
        return loc[0][0],loc[1][0]

    def run(self, ips, imgs, para = None):
        self.modedict={
        'SQDIFF':cv2.TM_SQDIFF,
        'SQDIFF_NORMED':cv2.TM_SQDIFF_NORMED,
        'CCORR':cv2.TM_CCORR,
        'CCORR_NORMED':cv2.TM_CCORR_NORMED,
        'CCOEFF':cv2.TM_CCOEFF
        }
        data = []
        sly, slx = ips.get_rect()
        #将矩形中的线进行等分，x是固定的
        print(sly,slx)
        img_moudle = imgs[0][sly, slx.start+60:slx.stop-60]
        n=len(imgs)
        locs = []
        for i in range(n):
            prgs=(i,n)
            self.progress(i, len(imgs))
            x,y=self.mathc_img(imgs[i][sly, slx.start:slx.stop],img_moudle,self.modedict[self.para['mode']])
            temp=imgs[i][self.para['amp_x']+(x-self.para['amp_x']):-self.para['amp_x']+(x-self.para['amp_x']),self.para['amp_y']+(y-self.para['amp_y']):-self.para['amp_y']+(y-self.para['amp_y'])]
            imgs[i][self.para['amp_x']:-self.para['amp_x'],self.para['amp_y']:-self.para['amp_y']]=temp
            locs.append((x, y))
        # print(locs)
        # IPy.show_table(pd.DataFrame(locs, ['X','Y']),'locations')
        IPy.show_table(pd.DataFrame(locs, columns=['X','Y']),'locations')
            
plgs = [Combine, Dark, '-', Gradient, DOG, Watershed,'-', Predict,Eliminate]
