import cv2
import numpy as np


class Efx:
    def __init__(self,imgSrc):
        self.imgSrc = imgSrc

    def filpH(self,destPath,name):
        destImg = cv2.flip(self.imgSrc,0)
        self.save(destImg,destPath,name,"FlipH")

    def filpV(self,destPath,name):
        destImg = cv2.flip(self.imgSrc,1)
        self.save(destImg,destPath,name,"FlipV")

    def grayScale(self,destPath,name):
        destImg = cv2.cvtColor(self.imgSrc,cv2.COLOR_BGR2GRAY)
        self.save(destImg,destPath,name,"Grayscale")

    def blur(self,destPath,name):
        destImg = cv2.GaussianBlur(self.imgSrc,(3,3),0)
        self.save(destImg,destPath,name,"blur")

    def filter_brightnes_contrast(self,destPath,name,brightness,contrast):
        """g(x) = contrast*f(x) + brightness"""
        hsv = cv2.addWeighted(self.imgSrc,contrast,np.zeros(self.imgSrc.shape,self.imgSrc.dtype),0,brightness)
        t = "[Brightness,Contrast]_[" + str(brightness)+ "," + str(contrast) + "]"
        # cv2.imwrite(destPath+"/" + name + t + ".png",hsv)
        self.save(hsv,destPath,name,t)


    def getXY(self,img,side):
        xf = 0
        yf = 0
        if side is 'tl':
            ry = range(0,img.shape[0]) 
            rx = range(0,img.shape[1])
        elif side is 'tr':
            ry = range(0,img.shape[0]) 
            rx = reversed(range(0,img.shape[1]))
            xf = img.shape[1] - 1
        elif  side is 'bl':
            ry = reversed(range(0,img.shape[0]))   
            rx = range(0,img.shape[1])
            yf = img.shape[0] - 1   
        elif  side is 'br':
            ry = reversed(range(0,img.shape[0])) 
            rx = reversed(range(0,img.shape[1]))
            yf = img.shape[0] - 1   
            xf = img.shape[1] - 1       
        else:
            return None

        cy = 0
        cx = 0

        for y in ry:
            px = img[y,xf]
            if px[0] != 0 or px[1] != 0 or px[2] != 0:
                cy = y
                break    
        for x in rx:
            px = img[yf,x]
            if px[0] != 0 or px[1] != 0 or px[2] != 0:
                cx = x
                break
        return (cx,cy)

    def getCoords(self, img):
        """
        return [tl,tr,bl,br]
        
        rl,bl,... = (x,y)
        """
        tl = self.getXY(img,'tl')
        tr = self.getXY(img,'tr')
        bl = self.getXY(img,'bl')
        br = self.getXY(img,'br')   

        return [tl,tr,bl,br]

    def getBndbox(self,img):
        c = self.getCoords(img)
        tl = c[0]
        tr = c[1]
        bl = c[2]
        br = c[3]

        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0

        # hr
        if tl[0] < tl[1]:
            x1 = tl[0]
            y1 = tr[1]

            x2 = br[0]
            y2 = bl[1]
        else:
            x1 = bl[0] 
            y1 = tl[1]

            x2 = tr[0]
            y2 = br[1]

        return (x1,y1),(x2,y2)

    def remove_black(self,img):
        tl,br = self.getBndbox(img)
        img = img[tl[1]:br[1] , tl[0]:br[0]]
        return img

    def rotate(self,destPath,name,angle):
        rows,cols,ch = self.imgSrc.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        destImg = cv2.warpAffine(self.imgSrc,M,(cols,rows))
        destImg = self.remove_black(destImg)
        self.save(destImg,destPath,name,"rotate[" + str(angle) +"]")

    def save(self,img,dest,name,opr):
        cv2.imwrite(dest+"/"+ name + "_" + opr + ".png",img) 