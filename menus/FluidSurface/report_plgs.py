from imagepy import IPy, wx
import numpy as np
from imagepy.core.engine import Simple, Filter
from imagepy.core.draw.paint import Paint
import cv2
import pandas as pd
class Mark:
    def __init__(self, xs, data):
        self.xs, self.data = xs, data

    def draw(self, dc, f, **key):
        dc.SetPen(wx.Pen((0,255,0), width=1, style=wx.SOLID))
        ys = self.data[0 if len(self.data)==1 else key['cur']]
        pts = zip(self.xs, ys)
        for i in pts:
            pos = f(*i)
            dc.DrawCircle(pos[0], pos[1], 2)
            #dc.DrawText('id={}'.format(i), pos[0], pos[1])

# center, area, l, extent, cov
class Surface(Simple):
    title = 'Find Surface'
    note = ['8-bit', 'req_roi']
    para = {'num':10}
    view = [(int, 'num', (3,30), 0, 'count', '')]

    #process
    def run(self, ips, imgs, para = None):
        data = []
        sly, slx = ips.get_rect()
        xs = np.linspace(0, slx.stop-slx.start-1, para['num']).astype(int)
        for img in imgs:
            img = img[sly, slx]
            ys = np.array([np.where(img[:,i]==255)[0].max() for i in xs])
            data.append(ys+sly.start)
        xs += slx.start
        ips.mark = Mark(xs, data)
        k,unit = ips.unit
        data = (np.array(data)*k).round(3)
        # IPy.table(ips.title+'-pts', data, ['%.3f'%i for i in xs*k])
        IPy.show_table(pd.DataFrame(data, columns=['%.3f'%i for i in xs*k]), ips.title+'-pts')

class DrawMark(Filter):
    title = 'Mark Surface'
    note = ['8-bit', 'req_roi']
    para = {'num':10}
    view = [(int, 'num', (3,30), 0, 'count', '')]

    #process
    def run(self, ips, snap, img, para = None):
        data = []
        sly, slx = ips.get_rect()
        xs = np.linspace(0, slx.stop-slx.start-1, para['num']).astype(int)
        painter = Paint()
        img2 = img[sly, slx]
        ys = np.array([np.where(img2[:,i]==255)[0].max() for i in xs])
        img2[cv2.dilate((img2==255).astype(np.uint8), np.ones((3,3)))>0] = 255
        for p in zip(ys+sly.start, xs+slx.start):
            painter.draw_point(img, p[1], p[0], 7, 255)
            

plgs = [Surface, DrawMark]