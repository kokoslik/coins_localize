import matplotlib.pyplot as plt
import numpy as np
import cv2

def calc_res(rect,contour):
    s=0
    for i in contour:
        x,y=i[0][0],i[0][1]
        x+=rect[0][0]
        y+=rect[0][1]
        x,y=x*np.cos(rect[2])-y*np.sin(rect[2]),x*np.sin(rect[2])+y*np.cos(rect[2])
        s+=(x/(rect[1][0]/2))**2+(y/(rect[1][1]/2))**2-1
    return np.sqrt(s/len(contour))

def check_ellipse(contour,threshhold):
    if len(contour)>5:
        rect=cv2.fitEllipse(contour)
        if len(contour)>0.8*(rect[1][0]+rect[1][1])/2*np.pi and len(contour) < 13*(rect[1][0]+rect[1][1])/2*np.pi:
            if calc_res(rect,contour)<threshhold:
                return True



def main():
    frame = cv2.imread('coins_easy.jpg')
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray,(5,5),0)
    high,temp = cv2.threshold(frame_gray,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    low=high*0.5
    edges = cv2.Canny(frame_gray,low,high,L2gradient=True)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)
    #edges=cv2.adaptiveThreshold(frame_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,7)
    #edges=cv2.Laplacian(frame_gray,cv2.CV_64F)
    #ret,edges=cv2.threshold(frame_gray,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fig = plt.figure(1)
    plt.imshow(dilation,cmap='gray')
    #fig.show()
    im, contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    good_contours=[]
    for i in contours:
        if check_ellipse(i,50):
            rect = cv2.fitEllipse(i)
            cv2.ellipse(frame,rect,(255,0,0),thickness=3)
    #cv2.drawContours(frame,contours,-1,(0,0,255),2)
    fig2 = plt.figure(2)
    plt.imshow(frame)
    plt.show()
    #input()

if __name__=='__main__':
    main()