import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from skimage import exposure
from netG import CustomUnet, ResnetGenerator

class seat_gen:
    def __init__(self):
        self.rgb = torch.jit.load("weights/rbg720.pt")
        self.win = torch.jit.load("weights/window_rm.pt")
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')  # get device name: CPU or GPU
        self.netG = CustomUnet()

        state_dict = torch.load("weights/latest_net_G.pth")
        self.netG.load_state_dict(state_dict)
        self.netG.to(self.device)

        self.cropnetG = CustomUnet()
        crop_state_dict = torch.load("weights/crop_latest_net_G.pth")
        self.cropnetG.load_state_dict(crop_state_dict)
        self.cropnetG.to(self.device)

        self.liveG = ResnetGenerator()
        live_dict = torch.load("weights/liveG.pth")
        self.liveG.load_state_dict(live_dict)
        self.liveG.to(self.device)
        

        
    def thumbnail(self, h, w, nh=1080):
        if h > nh:
            nw = int(nh*w/h)
            return nh, nw
        else:
            return h, w

    def cut_eval(self, img, size, crop='false'):
        shape = img.shape[:2]
        img = cv2.resize(img,size)[None]
        img = torch.from_numpy(img).cuda()
        img = img.permute(0,3,1,2).contiguous()
        img = img.div(127.5).sub_(-1)
        if crop == "false":
            pred = self.netG(img)
        elif crop == "true":
            pred = self.cropnetG(img)
        else:
            pred = self.liveG(img)
        pred = pred.add_(1.).mul_(127.5)
        pred = pred.to(torch.uint8).permute(0,2,3,1).contiguous()
        pred = pred.cpu().numpy()[0]
        pred = cv2.resize(pred,(shape[1],shape[0]))
        return pred

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

    def prep_win_full(self,p,oim):
        oim = Image.fromarray(oim)
        black = Image.new("RGB",oim.size,color=(255,255,255))
        bg = Image.new('RGB', oim.size,color=(127,127,127))
        bg.paste(black, (0, 0), oim)
        bg.paste(oim, (0,0), p)
        p = np.array(bg)
        return p

    def prep_win_crop(self,p,oim,blackout):
        x,y,w,h = cv2.boundingRect(np.array(p))
        x-=25
        y-=25
        w+=50
        h+=50
        oim = oim[y:y+h,x:x+w]
        oim = Image.fromarray(oim)
        black = Image.new("RGB",oim.size,color=(0,0,0))
        black.paste(oim,(0,0),oim)
        if blackout:
            p = p.crop((x,y,x+w,y+h))
            win = Image.new("RGB",p.size,color=(0,0,0))
            black.paste(win,(0,0),p)
        black = np.array(black)
        return black, x, y
        

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

    def remove_bg(self,oimg):
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
        return oimg2

    def remove_window(self,oimg2):
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
            pred2 = pred2.cpu().numpy()[0][0]
            oimg2 = oimg2.cpu().numpy()[0]
            pred2[pred2<128] = 0
        return pred2, oimg2

    def process_full(self,win_mask, oimg, rbg_img):
        prep1 = self.prep_win_full(win_mask,oimg)
        pred1 = self.cut_eval(prep1,(512,512))
        pred1 = Image.fromarray(pred1)
        rbg_img.paste(pred1,(0,0),win_mask)
        return rbg_img

    def process_crop(self,win_mask, oimg, rbg_img,blackout=False):
        prep1, x, y = self.prep_win_crop(win_mask,oimg,blackout)
        try:
            if blackout:
                pred1 = self.cut_eval(prep1,(320,320),crop='live')
            else:
                pred1 = self.cut_eval(prep1,(512,256),crop='true')
        except:
            return None
        pred1 = Image.fromarray(pred1)
        car = Image.new("RGB", rbg_img.size)
        car.paste(pred1,(x,y))
        rbg_img.paste(car,(0,0),win_mask)
        return rbg_img

    def process(self,oimg):
        rbg_img = self.remove_bg(oimg)
        win_mask, oimg = self.remove_window(rbg_img)
        rbg_img = Image.fromarray(rbg_img)
        win_mask = Image.fromarray(win_mask).convert("L")
        res1 = self.process_full(win_mask, oimg, rbg_img.copy())
        res2 = self.process_crop(win_mask, oimg, rbg_img.copy())
        res3 = self.process_crop(win_mask, oimg, rbg_img.copy(),blackout=True)
        if res2 is None or res3 is None:
            return None
        canvas = Image.new("RGB",size=(rbg_img.width*4, rbg_img.height))
        canvas.paste(rbg_img,(0,0))
        canvas.paste(res3,(rbg_img.width,0),res3)
        canvas.paste(res1,(rbg_img.width*2,0),res1)
        canvas.paste(res2,(rbg_img.width*3,0),res2)
        return canvas


class Test:
    def __init__(self):
        self.in_dir = "test_images"
        self.out_dir = "results"
        os.makedirs(self.out_dir, exist_ok = True)
        self.seat_gen = seat_gen()

    def __call__(self):
        for im in tqdm(os.listdir(self.in_dir)):
            im_path = os.path.join(self.in_dir,im)
            out_path = os.path.join(self.out_dir,im.split('.')[0]+'.png')
            if os.path.exists(out_path):
                continue
            img = np.array(Image.open(im_path))
            img = self.seat_gen.process(img)
            if img is None:
                print("window not detected for: ",im)
                continue
            img.save(out_path)

t = Test()
t()


        
