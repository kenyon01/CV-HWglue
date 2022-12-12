import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
class ImageDeal:
    def __init__(self,img):
        self.img = img
        self.img_gray=[]
        self.img_resized = []
        self.img_gaussian = []
        self.img_thresholds = []
        self.likeimage = []
        self.img_onlyleftCounters =[]
        self.img_onlyrightCounters = []
        self.img_onlyCounters=[]
        self.leftinnercontour = []
        self.rightinnercontour = []


    def resize(self,img):
        self.img_resized = cv2.resize(img, (800, 800), interpolation=cv2.INTER_CUBIC)
        return self.img_resized

    def gray_procession(self,img):
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.img_gray

    def threshold_procession(self,img, model_type, threshold_value):
        ret,self.img_thresholds = cv2.threshold(img, threshold_value, 255, model_type)
        # _, threshold_img = cv2.threshold(img, threshold, 10, cv2.THRESH_BINARY)
        # ret, threshold_img        = cv2.threshold(img, 100, 255,cv2.THRESH_BINARY)
        # _, threshold_img = cv2.threshold(img,160, 255, cv2.THRESH_TOZERO)
        return self.img_thresholds

    def gaussian_blur(self,img):
        self.img_gaussian = cv2.GaussianBlur(img, (5, 5), 10)
        return self.img_gaussian
    def newLikeImage(self):
        self.likeimage = np.zeros([self.img.shape[0], self.img.shape[1], self.img.shape[2]], np.uint8)
        # cv2.drawContours(likeimage, contours, -1, (255, 255, 255), 1)
        return self.likeimage
    def onlyCountersImage(self,contours):
        coors = np.array(contours)
        self.img_onlyCounters = np.zeros([self.img.shape[0], self.img.shape[1], self.img.shape[2]], np.uint8)
        for coor in coors:
            # print(coor)
            cv2.circle(self.img_onlyCounters, (int(coor[0]),int(coor[1])),1,[255,255,255],1)
    def onlySplitCountersImage(self,leftcontour,rightcontour):
        coors = np.array(leftcontour)
        self.img_onlyleftCounters = np.zeros([self.img.shape[0], self.img.shape[1], self.img.shape[2]], np.uint8)
        for coor in coors:
            # print(coor)
            cv2.circle(self.img_onlyleftCounters, (int(coor[0]),int(coor[1])),1,[255,255,255],1)
        coors = np.array(rightcontour)
        self.img_onlyrightCounters = np.zeros([self.img.shape[0], self.img.shape[1], self.img.shape[2]], np.uint8)
        for coor in coors:
            # print(coor)
            cv2.circle(self.img_onlyrightCounters, (int(coor[0]), int(coor[1])), 1, [255, 255, 255], 1)
        return self.img_onlyleftCounters,self.img_onlyrightCounters
    def selectTwoRound(self,leftcontour,rightcontour,leftCOC,rightCOC,pts):

        # leftcontour = ROISelectionModel.leftcontour
        # rightcontour = ROISelectionModel.rightcontour
        splitpoint = (pts[1][0]+pts[2][0])/2
        # leftcontour,rightcontour = self.splitContours(contours,splitpoint)
        self.leftinnercontour = self.getSingleCicle(leftcontour, leftCOC)

        self.rightinnercontour = self.getSingleCicle(rightcontour, rightCOC)
        return self.leftinnercontour, self.rightinnercontour
    def splitContours(self,picked_data,splitpoint):
        # splitpoint = self.findSplitPoint(ROISelectionModel)
        leftcontour = []
        for coor in picked_data:
            if coor[0] < splitpoint:
                leftcontour.append(coor)
        rightcontour = []
        for coor in picked_data:
            if coor[0] > splitpoint:
                rightcontour.append(coor)
        return leftcontour,rightcontour
    def splitCOC(self,COC):
        leftcoc, rightcoc = self.sortCOC(COC)
        return leftcoc,rightcoc
    def findSplitPoint(self,Model):
        minvalue = min(np.array(pickdata)[:,1])
        maxvalue = max(np.array(pickdata)[:,1])
        splitpoint = (maxvalue-minvalue)/2
        return splitpoint
    def sortCOC(self,COC):
        A = COC[0]
        B = COC[1]
        if A[0] > B[0]:
            B,A = A,B
        return A,B
    def getSingleCicle(self,Data, center):
        Temp = []
        for i in range(-180, 180, 1):
            PointAtCurrentAngle, _ = self.traverseData(Data, center, i)
            contour = self.selectSiglPoint(PointAtCurrentAngle, center)
            if contour is not None:
                Temp.append(contour)
            # Temp = removeOutlier(Temp)
        return Temp
    def selectSiglPoint(self,PointAtCurrentAngle, v1):
        Dis = []
        for j in range(len(PointAtCurrentAngle)):
            Dis.append(np.linalg.norm(np.array(PointAtCurrentAngle[j]) - np.array(v1)))
        if Dis != []:
            return PointAtCurrentAngle[Dis.index(min(Dis))]
    def removeOutlier(BestPoint):
        res = list(filter(None, BestPoint))
        return res
    def traverseData(self, Data, center, angle):
        PointAtCurrentAngle = []
        res_memery = []
        Data = np.array(Data)
        center = np.array(center)
        for coor in Data:
            res = coor - center
            res = res.tolist()
            res_memery.append(res)
            # print(res)
            angle1 = self.angle_of_vector(res)
            if abs(angle1 - angle) < 0.1:
                if np.linalg.norm(coor-center)< 250 and np.linalg.norm(coor-center)>100:
                    print('angle:', angle)
                    temp = coor.tolist()
                    PointAtCurrentAngle.append(temp)
        return PointAtCurrentAngle, res_memery
    def angle_of_vector(self,v1):
        angle1 = math.atan2(v1[0], v1[1])
        angle1 = int(angle1 * 180 / math.pi)
        return angle1

    def Deal(self):
        pass
