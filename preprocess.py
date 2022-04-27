import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from skimage import exposure
import shutil
class preprocess:
    def __init__(self,mode):
        self.rgb = torch.jit.load("weights/rbg720.pt")
        self.win = torch.jit.load("weights/window_rm.pt")
        self.mode = mode
        
    def thumbnail(self, h, w, nh=1080):
        if h > nh:
            nw = int(nh*w/h)
            return nh, nw
        else:
            return h, w

    def post_rgb(self, p, oim):
        oim = Image.fromarray(oim)
        p = self.remove_outside_areas(p[0][0])
        p = Image.fromarray(p)
        bbox = p.getbbox()
        p = p.crop(bbox)
        oim = oim.crop(bbox)
        bg = Image.new('RGBA', oim.size)
        bg.paste(oim, (0, 0), p)
        p = np.array(bg)
        return p

    def post_win_full(self,p,oim):
        p = p[0][0]
        p[p<128] = 0        
        oim = Image.fromarray(oim)
        p = Image.fromarray(p)
        black = Image.new("RGB",oim.size,color=(255,255,255))
        bg = Image.new('RGB', oim.size,color=(127,127,127))
        bg.paste(black, (0, 0), oim)
        bg.paste(oim, (0,0), p)
        p = np.array(bg)
        return p

    def post_win_crop(self,p,oim):
        p = p[0][0]
        p[p<128] = 0        
        x,y,w,h = cv2.boundingRect(p)
        # print(x,y,w,h)
        x-=25
        y-=25
        w+=50
        h+=50
        oim = oim[y:y+h,x:x+w]
        oim = Image.fromarray(oim)
        # p = Image.fromarray(p)
        black = Image.new("RGB",oim.size,color=(0,0,0))
        black.paste(oim,(0,0),oim)
        black = np.array(black)
        return black

    def remove_outside_areas(self, mask):
        mask[mask < 128] = 0
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_areas = [cv2.contourArea(cnt) for cnt in contours]
        max_area = max(contours_areas)
        for cnt, area in zip(contours, contours_areas):
            if area < max_area:
                cv2.drawContours(mask, [cnt], -1, (0, 0, 0), -1)
            elif area == max_area:
                cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.5, sigmaY=1.5, borderType=cv2.BORDER_DEFAULT)
        mask = exposure.rescale_intensity(mask, in_range=(180, 255), out_range=(0, 255)).astype(np.uint8)
        return mask

    def process(self,oimg):
        img = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (720, 720), interpolation=cv2.INTER_CUBIC)
        img = torch.from_numpy(np.array([img]))#.permute(0, 3, 1, 2)
        oimg = torch.from_numpy(np.array([oimg]))#.permute(0, 3, 1, 2)
        with torch.inference_mode():
            img, oimg = img.cuda(), oimg.cuda()
            pred = self.rgb(img)
            _, h, w, _ = oimg.shape
            h, w = self.thumbnail(h, w, 1080)
            oimg = oimg.permute(0, 3, 1, 2)
            oimg = F.interpolate(oimg.float(), (h, w), mode='bilinear').to(torch.uint8)
            pred = F.interpolate(pred.float(), (h, w), mode='bilinear').to(torch.uint8)
            oimg = oimg.permute(0, 2, 3, 1)
            pred = pred.cpu().numpy()
            oimg = oimg.cpu().numpy()[0]
        oimg2 = self.post_rgb(pred,oimg)
        img2 = cv2.resize(oimg2, (768, 768), interpolation=cv2.INTER_CUBIC)[...,:3]
        img2 = torch.from_numpy(np.array([img2]))
        oimg2 = torch.from_numpy(np.array([oimg2]))
        with torch.inference_mode():
            img2, oimg2 = img2.cuda(), oimg2.cuda()
            pred2 = self.win(img2)
            _, h, w, _ = oimg2.shape
            h, w = self.thumbnail(h, w, 1080)
            oimg2 = oimg2.permute(0, 3, 1, 2)
            oimg2 = F.interpolate(oimg2.float(), (h, w), mode='bilinear').to(torch.uint8)
            pred2 = F.interpolate(pred2.float(), (h, w), mode='bilinear').to(torch.uint8)
            oimg2 = oimg2.permute(0, 2, 3, 1)
            pred2 = pred2.cpu().numpy()
            oimg2 = oimg2.cpu().numpy()[0]
        if self.mode == "full":
            return self.post_win_full(pred2,oimg2)
        return self.post_win_crop(pred2,oimg2)
        
class prep_images:
    def __init__(self,mode):
        self.in_dir = "test_images"
        self.out_dir = "prep_images/testA"
        self.out_dir_orig = "prep_images/testB"
        os.makedirs(self.out_dir, exist_ok = True)
        os.makedirs(self.out_dir_orig, exist_ok = True)
        self.preprocess = preprocess(mode)

    def __call__(self):
        for im in tqdm(os.listdir(self.in_dir)):
            im_path = os.path.join(self.in_dir,im)
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            img = self.preprocess.process(img)
            if img.size == 0:
                print("Window not detected for ",im)
                continue
            cv2.imwrite(os.path.join(self.out_dir,im.split('.')[0]+'.png'),img)
            shutil.copy(im_path,os.path.join(self.out_dir_orig,im))

t = prep_images("full")
t()


        
