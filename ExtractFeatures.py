# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:55:33 2015

@author: Aaron
"""

# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob

#change your root path
listOfPaths = ["/Users/Aaron/Documents/TwentyFifteen/Winter 2015/EECS 395- Machine Learning/Course Project/Sample Images/1/*.JPG",
               "/Users/Aaron/Documents/TwentyFifteen/Winter 2015/EECS 395- Machine Learning/Course Project/Sample Images/2/*.JPG",
               "/Users/Aaron/Documents/TwentyFifteen/Winter 2015/EECS 395- Machine Learning/Course Project/Sample Images/3/*.JPG",
               "/Users/Aaron/Documents/TwentyFifteen/Winter 2015/EECS 395- Machine Learning/Course Project/Sample Images/4/*.JPG",
               "/Users/Aaron/Documents/TwentyFifteen/Winter 2015/EECS 395- Machine Learning/Course Project/Sample Images/5/*.JPG",
               "/Users/Aaron/Documents/TwentyFifteen/Winter 2015/EECS 395- Machine Learning/Course Project/Sample Images/6/*.JPG",
               "/Users/Aaron/Documents/TwentyFifteen/Winter 2015/EECS 395- Machine Learning/Course Project/Sample Images/7/*.JPG",
               "/Users/Aaron/Documents/TwentyFifteen/Winter 2015/EECS 395- Machine Learning/Course Project/Sample Images/8/*.JPG",
               "/Users/Aaron/Documents/TwentyFifteen/Winter 2015/EECS 395- Machine Learning/Course Project/Sample Images/9/*.JPG",
               "/Users/Aaron/Documents/TwentyFifteen/Winter 2015/EECS 395- Machine Learning/Course Project/Sample Images/10/*.JPG"]
 

def fillholes(gray):
    #Function to fill holes in a gray (or threshed) image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    res = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)

ArrayToBeExported= [0,0,0,0,0,0] # initializing, this row to be deleted later

for i in range(1,10):
    label = i
    mypath = listOfPaths[i-1]
    files = glob.glob(mypath)
    for file in files:
        #Read in image, set to gray and then threshold
        InputImage = cv2.imread(file, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(InputImage, cv2.COLOR_BGR2GRAY)
        ret,fg = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY_INV)
        #fg = cv2.Canny(fg, 127,255)
        #Elimanting noise
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        fg= cv2.erode(fg,element)
        fillholes(fg)
        
        
        #Finding the contours
        contours, hierarchy = cv2.findContours(fg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        #Declaring variables to begin finding largest contour
        max_area = 0
        ci = 0
        len_contours = len(contours)
    
        if len_contours != 0: # if contours found
            cnt = contours[0]
            #Now about to find the largest contour (the hand)
            for i in xrange(len(contours)):
                cnt=contours[i]
                area = cv2.contourArea(cnt)
                if(area>max_area):
                    max_area=area
                    ci=i
                cnt=contours[ci]
        
            #Declaring hull & initializing the drawing image
            hull = cv2.convexHull(cnt)
            #Drawing contours and hull (contour follows hand, hull is outside)
            drawing = cv2.cvtColor(np.copy(fg), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(drawing,[cnt],0,(255,255,0),2)
            cv2.drawContours(drawing,[hull],0,(100,100,150),2)
            
            #resetting Hull and setting contours
            cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            hull = cv2.convexHull(cnt, returnPoints = False)
            
            
            ## GETTING ANGLE
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
            #rect = cv2.minAreaRect(cnt) # don't need this anyore, using the fitEllipse for angle, but just leaving it in for reference
            #getting area of contour
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
        
           #Initializing moments: NEVER ACTUALLY USE THIS, but leaving it in in case we want to in the future
            centr=(0,0)#Declaring Center
            moments = cv2.moments(cnt)
            if moments['m00']!=0:
                #Now finding the center of this hull
                cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                cy = int(moments['m01']/moments['m00']) # cy = M01/M00              
                centr=(cx,cy) 
                cv2.circle(drawing,centr,5,[0,0,255],2) #Drawing the center of the hand in the image
            
            #finding fingertips
    
            NumberOfDetectedFingers=0;
            if(hull.shape[0] > 3 and cnt.shape[0] > 3): # Don't know if we need this test, seems unnecessary
                defects = cv2.convexityDefects(cnt,hull) # getting the convexity defects
                y_points = []
                if not defects is None:
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(cnt[s][0]) # the start of the convexity defect(when the contour curves in)
                        y_points.append(start)
            
                    for point in y_points:
                        cv2.circle(drawing,point,5,[255,0,0],-1) # drawing circles on the fingertips
                NumberOfDetectedFingers= len(y_points)
            
            
    
            #creating vector of features
            #feature_point = [1,angle,NumberOfDetectedFingers,label]
            # IF WE WANT ADDITIONAL FEATURES.. (area and perimeter)         
            feature_point = [1,angle,NumberOfDetectedFingers,area, perimeter,label]
            
            ##LOADING DATA INTO ARRAY TO BE EXPORTED   
            ArrayToBeExported= np.vstack([ArrayToBeExported,feature_point]) # adding each feature row to the array
    
            
            #Show image
            drawing2 = cv2.resize(drawing,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
            cv2.imshow('drawing', drawing2) #comment this out when we're actually doing the feature extraction, just for visual debugging purposes            
            cv2.waitKey(0)
            cv2.destroyAllWindows()

ArrayToBeExported = ArrayToBeExported[1:] # deleting the initializing row
ArrayToBeExported = np.asarray(ArrayToBeExported)
np.savetxt("ArrayToBeExported4F.csv", ArrayToBeExported, delimiter=",") # saves as csv
# IN MATLAB, can use M = csvread(filename) , or just navigate to the csv file and open it.