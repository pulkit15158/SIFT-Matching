
###########3Pulkt Goel 2015158
################################## Q1  ##################################################3 

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math



 ################### RAnsac#############################################
def Ransac(src_pts,dst_pts,iter=2500,treshold=50):
	error = np.zeros(iter) 
	M1 = []
	x = np.concatenate((src_pts,dst_pts),axis = 1)	
	for i in range(iter):
		idx = np.random.randint(len(src_pts), size=4)
		x_new = x[idx,:]
		src_pts_new = x_new[:,0]
		dst_pts_new = x_new[:,1]
		M1.append(my_Dlt_algo(src_pts_new,dst_pts_new))
		idx = np.random.randint(len(src_pts),size=len(src_pts)-4)
		pts= x[idx,:]
		pts_src = np.float32(pts[:,0]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts_src,M1[i])
		for k in range(len(pts)):
			error[i] = error[i] + math.sqrt((pts[k][1][0]-dst[k][0][0])**2 + (pts[k][1][1]-dst[k][0][1])**2)			
	#print error,error[error.argmin()]	
	final_matrix = M1[error.argmin()]
	pts= x
	mask = []	
	pts_src = np.float32(pts[:,0]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts_src,final_matrix)
	for k in range(len(pts)):
		#print math.sqrt((pts[k][1][0]-dst[k][0][0])**2 + (pts[k][1][1]-dst[k][0][1])**2) 	
		if(math.sqrt((pts[k][1][0]-dst[k][0][0])**2 + (pts[k][1][1]-dst[k][0][1])**2) < treshold):
			mask.append(1)
		else:
			mask.append(0)		
	return final_matrix,mask	

############################# DLT ALGO ##################################################3

def my_Dlt_algo(src_pts,dst_pts):
	a = np.zeros((2*len(src_pts),9))
	for i in range(len(src_pts)):
		a[2*i] = [-src_pts[i][0],-src_pts[i][1],-1,0,0,0,dst_pts[i][0]*src_pts[i][0],dst_pts[i][0]*src_pts[i][1],dst_pts[i][0]]
		a[2*i+1] = [0,0,0,-src_pts[i][0],-src_pts[i][1],-1,dst_pts[i][1]*src_pts[i][0],dst_pts[i][1]*src_pts[i][1],dst_pts[i][1]]
	u,s,v = np.linalg.svd(a)
	h = v[8].reshape((3,3))
	s = np.linalg.svd(a)
	return h/h[2,2]	

img1 = cv2.imread('test2.jpg',0)          # queryImage
img2 = cv2.imread('collage_image.jpg',0) # trainImage

########################## Shift script taken from opencv #########################################################3

sift = cv2.xfeatures2d.SIFT_create()

######################### finding  the keypoints and descriptors with SIFT ###################################

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv2.BFMatcher() # matching keypoints using BFmatcher
matches = bf.knnMatch(des1,des2, k=2)
matchesMask = [[0,0] for i in xrange(len(matches))]

good = []
dist = 0.9
while(len(good)<200):
	for m,n in matches:
		if m.distance < dist*n.distance:
			good.append(m)
	dist = dist+0.0001
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

imgc_1 = cv2.imread('test2.jpg')          # queryImage
imgc_2 = cv2.imread('collage_image.jpg') # trainImage

for i in range(len(src_pts)):
	cv2.line(imgc_1,(src_pts[i][0][0],src_pts[i][0][1]),(src_pts[i][0][0],src_pts[i][0][1]),(0,0,255),5)
	cv2.line(imgc_2,(dst_pts[i][0][0],dst_pts[i][0][1]),(dst_pts[i][0][0],dst_pts[i][0][1]),(0,0,255),5)
#img4 = cv2.drawMatches(img1,src_pts)
imgc_1 = imgc_1[:,:,::-1]
plt.imshow(imgc_1),plt.show()
imgc_2 = imgc_2[:,:,::-1]
plt.imshow(imgc_2),plt.show()


h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2) # taking corner points

M1,matchesMask = Ransac(src_pts,dst_pts,2000,10) ########### appling Ransac and DLT algo
print M1

dst= cv2.perspectiveTransform(pts,M1)  
      
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

####################3 plotting final images ######################################
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
p1 = (dst[0][0][0],dst[0][0][1])
p2 = (dst[1][0][0],dst[1][0][1])
p3 = (dst[2][0][0],dst[2][0][1])
p4 = (dst[3][0][0],dst[3][0][1])
img2 = cv2.imread('collage_image.jpg')
cv2.line(img2,p1,p2,(0,0,255),15)
cv2.line(img2,p2,p3,(0,0,255),15)
cv2.line(img2,p3,p4,(0,0,255),15)
cv2.line(img2,p4,p1,(0,0,255),15)
img2 = img2[:,:,::-1]
plt.imshow(img3,),plt.show()
plt.imshow(img2),plt.show()









######################################## Q2 ##########################################
from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

xy = []

################ Function for clicking points of image 
def corxy(event, x, y, flags, param):
	global xy
	if event ==cv2.EVENT_LBUTTONDOWN:
		print x,y
		xy.append((x,y,1)) # converting to homo
img1 = cv2.imread('floor.jpg',0)          # queryImage
cv2.namedWindow('click')
cv2.setMouseCallback('click',corxy)
cv2.imshow('click',img1)
cv2.waitKey(0)
xy = np.array(xy)

################## Affine Rectification ######################
l1 = np.cross(xy[0],xy[1])
l2 = np.cross(xy[2],xy[3])
m1 = np.cross(xy[0],xy[2])
m2 = np.cross(xy[1],xy[3])
v1 = np.cross(l1,l2)
v2 = np.cross(m1,m2)
v = np.cross(v1,v2)
print v[0],v[1],v[2]
# v[0] = float(v[0])/float(v[2])
# v[1] = float(v[1])/float(v[2])
# v[2] = 1.0

# print v[0],v[1]
h1 = [[1,0,0],[0,1,0],[v[0]/v[2],v[1]/v[2],1.0]]
xy = np.array(xy)
l1 = np.cross(xy[0],xy[1])
l2 = np.cross(xy[2],xy[3])
m1 = np.cross(xy[0],xy[2])
m2 = np.cross(xy[1],xy[3])
v1 = np.cross(l1,l2)
v2 = np.cross(m1,m2)
v = np.cross(v1,v2)
print h1
h,w = img1.shape
warp = cv2.warpPerspective(img1, np.array(h1), (h,w))				
cv2.namedWindow('image')
cv2.imshow('image',warp)
cv2.waitKey(0)
cv2.destroyAllWindows()


############### Metric Rectification ###########################

xy_new = []
for i in range(len(xy)):
	xy_new.append([xy[i][0],xy[i][1]]) 
pts = np.float32(xy_new).reshape(-1,1,2)
dst_1 = cv2.perspectiveTransform(pts,np.array(h1))
dst = []
for i in range(len(pts)):
	dst.append((dst_1[i][0][0],dst_1[i][0][1],1.0))
print dst
l1 = np.cross(dst[0],dst[1])
l2 = np.cross(dst[0],dst[3])
m1 = np.cross(dst[0],dst[2])
m2 = np.cross(dst[2],dst[1])

matrix = [[l1[0]*m1[0],l1[0]*m1[1]+l1[1]*m1[0]],[l2[0]*m2[0],l2[0]*m2[1]+l2[1]*m2[0]]]
# matrix = np.array(matrix).reshape((2,2))
b = [-l1[1]*m1[1],-l2[1]*m2[1]]
b = np.array(b).reshape(2,1)
print matrix,b
x = np.linalg.solve(matrix, b)
s = [[x[0][0],x[1][0]],[x[1][0],1.0]]
print s
u,d,v = np.linalg.svd(s)
d = [[math.sqrt(d[0]),0],[0,math.sqrt(d[1])]]
u = np.transpose(u)
a = np.matmul(u,d)
a = np.matmul(v,a)
h2 = [[a[0][0],a[0][1],0.0],[a[1][0],a[1][1],0.0],[0.0,0.0,1.0]]
h2 = np.linalg.inv(h2)
h,w = warp.shape
warp = cv2.warpPerspective(warp,h2, (h,w))				
cv2.namedWindow('image')
cv2.imshow('image',warp)
cv2.waitKey(0)
cv2.destroyAllWindows()