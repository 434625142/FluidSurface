from imagepy.core.engine import Filter, Simple
from imagepy.ipyalg import watershed
import numpy as np
import cv2
from keras.models import load_model
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
    view = [(float, (0,30), 1,  'sigma', 'sigma', 'pix')]

    #process
    def run(self, ips, snap, img, para = None):
        l = int(para['sigma']*3)*2+1
        cv2.GaussianBlur(snap, (l, l), para['sigma'], dst=img)
        msk = img<snap
        img-=snap
        img[msk] = 0

class Watershed(Filter):
    title = 'Watershed Surface'
    note = ['8-bit', 'auto_snap', 'not_channel', 'preview']

    #process
    def run(self, ips, snap, img, para = None):
        markers = img*0
        markers[[0,-1]] = [[1],[2]]
        mark = watershed(img, markers, line=True, conn=1)
        img[:] = (mark==0) * 255
class predict(Filter):
    title = 'predict'
    note = ['8-bit', 'auto_snap', 'not_channel', 'preview']
    mode_list=['msk','line','line on ori']
    view = [
                 (list, mode_list, str, 'mode', 'mode', '')
            ]
    para = {'model':load_model('plugins/434625142~FluidSurface/menus/FluidSurface/U-net.h5'),
            'mode':mode_list[0]
    }
    #一定要预测一次，否则后面会出错
    print(para['model'].predict(np.zeros((1, 224,224,1))))
    def run(self, ips, snap, img, para = None,):
        img[:]=self.my_predict(img,para['model'],para)
    def my_predict(self,img,model,para):
        shape_temp=img.shape
        img_temp=img.copy()
        img=cv2.resize(img,(224,224)).reshape(1,224,224,1).astype('float32')/255.0
        pred=(model.predict(img[:,:,:,])*255).astype('uint8').reshape(224,224)
        img=cv2.resize(pred,(shape_temp[1],shape_temp[0]))
        ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        if para['mode']=='msk':return thresh1 
        uint8_x = cv2.convertScaleAbs(cv2.Sobel(thresh1,cv2.CV_16S,1,0))
        uint8_y = cv2.convertScaleAbs(cv2.Sobel(thresh1,cv2.CV_16S,0,1))
        sobel_img = cv2.addWeighted(uint8_x,0.5,uint8_y,0.5,0)
        ret,line = cv2.threshold(sobel_img,127,255,cv2.THRESH_BINARY)
        if para['mode']=='line':return line 
        elif para['mode']=='line on ori':       
            img_temp[line>0]=255
            return img_temp

plgs = [Combine, Dark, '-', DOG, Watershed,'-',predict]
