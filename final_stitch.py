# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:25:08 2021

@author: Dell

https://sandipanweb.wordpress.com/2017/05/16/some-more-computational-photography-merging-and-blending-images-using-gaussian-and-laplacian-pyramids-in-python/
http://graphics.cs.cmu.edu/courses/15-463/2010_spring/Lectures/blending.pdf
https://github.com/cynricfu/multi-band-blending
https://courses.engr.illinois.edu/cs498dh/fa2011/lectures/Lecture%2018%20-%20Photo%20Stitching%20-%20CP%20Fall%202011.pdf
http://www.cse.psu.edu/~rtc12/CSE486/lecture06.pdf
"""
import harris
import cv2
import matplotlib.pyplot as plt
import math
import skimage.filters
import numpy as np
import skimage.io as sk
import random

def harris_points(images):
    img_response=[]
    final_coord=[]
        
    for img in images:
        rgb_img,gray_img = img
        h_r, img_points = harris.get_harris_corners(gray_img)
        print(h_r.shape,img_points.shape)
        x_new,y_new,coord = adaptive_suppression(img_points, h_r)
        #print((h_r),type(img_points),type(x),type(coord))
        # implot = plt.imshow(rgb_img)
        # Red dots of size 40
        # plt.scatter(x=img_points[1], y=img_points[0], c='r', s=10)
        # plt.show()
        
        # implot = plt.imshow(rgb_img)
         #Red dots of size 40
        # plt.scatter(x=x_new, y=y_new, c='r', s=10)
        # plt.show()
    
        final_coord.append(coord)
        #print(final_coord,len(coord))
        img_response.append(h_r)
        
    descriptors = []
    harris_pts = []
    for i in range(len(final_coord)):
        hp, h = final_coord[i], img_response[i]
        rgb_img, gray_img = images[i]

        feature_descriptors, h_pts = feature_descriptor(hp, rgb_img, gray_img, gray_img.shape)
        print(h_pts)
        pts1_x, pts1_y = [], []
        for pt in h_pts:
            x, y = pt
            pts1_x.append(x)
            pts1_y.append(y)
        implot3 = plt.imshow(rgb_img)
        plt.scatter(x=pts1_x, y=pts1_y, c='r', s=10)
        plt.show()
        
        descriptors.append(feature_descriptors)
        harris_pts.append(h_pts)
    
    match_pt, feature_desc, h_point1, h_point2 = feature_match(descriptors[1],descriptors[0],harris_pts[1],harris_pts[0])
    image1 = images[0][0]
    image2 = images[1][0]    
    """
    for i in range(5):
        
        feature_arr = feature_desc[i]
        match = match_pt[i]
        f1, f2 = feature_arr
        match1, match2 = match
        implot_1 = plt.imshow(image1)
        plt.scatter(x=match1[0], y=match1[1], c='r', s=40)
        plt.show()
        sk.imshow(f1)
        sk.show()
        implot_2 = plt.imshow(image2)
        plt.scatter(x=match2[0], y=match2[1], c='r', s=40)
        plt.show()
        sk.imshow(f1)
        sk.show()
        
        # Split points into x, y groups
        pts1_x, pts1_y = [], []
        for pt in h_point1:
            x, y = pt
            pts1_x.append(x)
            pts1_y.append(y)
        implot3 = plt.imshow(image1)
        plt.scatter(x=pts1_x, y=pts1_y, c='r', s=40)
        plt.show()

		# Split points into x, y groups
        pts2_x, pts2_y = [], []
        for pt in h_point2:
            x, y = pt
            pts2_x.append(x)
            pts2_y.append(y)
        implot4 = plt.imshow(image2)
        plt.scatter(x=pts2_x, y=pts2_y, c='r', s=40)
        plt.show()
    """
    print(len(match_pt))
    H,inliers = ransac_calculate(match_pt,image2,image1)
    print(len(inliers),len(inliers)/(5.9+0.22*len(match_pt)))
    img2_warp,img1_warp,mat,c = warping(image2,image1,H,np.eye(3))
    print(img2_warp.shape[0],img1_warp.shape[1])
    #dst = np.zeros((img2_warp.shape[1],img2_warp.shape[0]),dtype=np.int)
    src1 = img1_warp.copy() 
    src2 = img2_warp.copy()
    overlap = img2_warp.copy()
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.title.set_text("Image1 Warped")
    i = plt.imshow(img2_warp)
    b = fig.add_subplot(1, 2, 2)
    b.title.set_text("Image2 Warped")
    j= plt.imshow(img1_warp)
    plt.show()
   
    for i in range(img2_warp.shape[0]-1):
        for j in range(img2_warp.shape[1]-1):
            #print(i,j)
            if i==img2_warp.shape[0] or j==img2_warp.shape[1]:
                print(i,j,"in")
                break
            if img2_warp[i,j][0]>0.0 or img2_warp[i,j][1]>0.0 or img2_warp[i,j][2]>0.0:
                if img1_warp[i,j][0]>0.0 or img1_warp[i,j][1]>0.0 or img1_warp[i,j][2]>0.0:
                #print("in1")
                    overlap[i,j]=255
                #src2[i,j][1]=255
                #src2[i,j][2]=255
                else:
                    overlap[i,j]=0
                src2[i,j]=255
            if img1_warp[i,j][0]>0.0 or img1_warp[i,j][1]>0.0 or img1_warp[i,j][2]>0.0:
                if img2_warp[i,j][0]>0.0 or img2_warp[i,j][1]>0.0 or img2_warp[i,j][2]>0.0:
                #print("in2")
                    overlap[i,j]=255
                else:
                    overlap[i,j]=0
                #src1[i,j][1]=255
                src1[i,j]=255
    l_x=len(np.where(overlap==255)[0])
    l_y=len(np.where(overlap==255)[1])
    o=np.where(overlap==255)
    #print(o[0][0],o[0][l-1],o[1][0],o[1][l-1],o[2][0],o[2][l-1],overlap[o[0][0],o[1][0]])
    """
    f = plt.figure()
    plt.imshow(overlap)
    plt.scatter(o[1][1000],o[0][1000],color='r')
    plt.scatter(o[1][l-1],o[0][l-1],color='r')
    plt.show()
    """
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.title.set_text("Image1 Warped")
    i = plt.imshow(src1)
    b = fig.add_subplot(1, 2, 2)
    b.title.set_text("Image2 Warped")
    j= plt.imshow(src2)
    plt.show()
    print(type(img1_warp))
    
    band_blend = BandBlend()
    mosaic_img = band_blend.multi_band_mosiac(img1_warp, img2_warp,100, True,c,overlap)
    cv2.imshow("Mosaic",mosaic_img)
    cv2.waitKey(0)
    """
    print(mosaic_img.shape)
    for i in range(l_x):
        for j in range(l_y):
            x=o[0][i]
            y=o[1][j]
            mosaic_img[x,y] = (img1_warp[x,y] + img2_warp[x,y])/2
    fig = plt.figure()
    plt.imshow(mosaic_img)
    plt.show()
    """
#    mosaic_img = mosaic_create(img2_warp, img1_warp, image2, image1, c)
    
    #cv2.imshow("Final",mosaic_img)
    #cv2.waitKey(0)
    """
    a,b,m = preprocess(img1_warp, img2_warp, 400, False)
    cv2.imshow("check",m)
    cv2.waitKey(0)
    
    mosaic_img = mosaic_create(img2_warp, img1_warp, image2, image1, c)
    #cv2.imwrite("F:\ML\Image_Color\mos.jpg",mosaic_img)
    fig=plt.figure()
    p = plt.imshow(mosaic_img)
    plt.show()
    cv2.imshow("Mosaic",mosaic_img)
    cv2.waitKey(0)
    """

        
def distance(x1, y1, x2, y2):
	distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
	return distance        
        
def adaptive_suppression(interest_pt,h_r,c_robust=0.9,n=500):
    """
    Values of c_robust and n are initialized as mentioned in "Multi-Image Matching using Multi-Scale Oriented Patches" by Brown et. al
    """
    result = []
    x_orig = interest_pt[1]
    y_orig = interest_pt[0]
    
    x_orig_len = len(x_orig)
    y_orig_len = len(y_orig)
    
	
	# corner strength comparison
    for i in range(x_orig_len):
        x_i, y_i = x_orig[i], y_orig[i]
        
        #Initializing infinity as suppression radius that will decrease further
        radius_i = float("inf")
        j_i = (-1, -1)
        
        #Loop till minimum suppression radius obtained for each interest point
        for j in range(y_orig_len):
            x_j, y_j = x_orig[j], y_orig[j]

            radius_cal = distance(x_i, y_i, x_j, y_j)
			# Applying condition for minimum suppression radius
            if (x_i, y_i) != (x_j, y_j) and h_r[y_i][x_i] < h_r[y_j][x_j] * c_robust and radius_cal < radius_i:
                radius_i = radius_cal
                j_i = (x_j, y_j)

        result.append([radius_i, j_i])
	
	# Sort list of interest points by descending radius
    #print(result[:10])
    result = sorted(result, key = lambda sub: (-sub[0]))
    #print(result[:10])
    result = result[::-1] 
    result = result[1:] # Exclude infinity
    x_new, y_new, points = [], [], []

	# Get points with highest supression radius
    for i in result[1:n]:
        if i[0] < float("inf"):
            x_new.append(i[1][0])
            y_new.append(i[1][1])
            points.append((i[1][0], i[1][1]))
    return x_new, y_new, points


def feature_descriptor(points, color_img, input_img, img_shape):
	
    """
	Implement section 4 of obtaining feature descriptors as mentioned in "Multi-Image Matching using Multi-Scale Oriented Patches" by Brown et. al
	"""
	
    #Initialize list for features(8X8), harris points and large patches(40X40)
    features=[]
    large_patch=[]
    harris_pt=[]
    # print(img_shape)
    for pt in points:
        #Initialize large window of 40X40 for each point
        lp_curr = np.zeros((40,40))
        #Initialize small window of 8X8 for each point
        sp_curr = np.zeros((8,8))
        i = 0

        
		# Obtaining point centered 40x40 pixel window
        for y in range(-20, 20):
            j = 0
            for x in range(-20, 20):
                x_pt = pt[1]+y
                y_pt = pt[0]+x
     #           print(img_shape,x_pt, y_pt)
                if x_pt>=img_shape[0]:
      #              print("x",x_pt)
                    x_pt = img_shape[0]-1
                if y_pt>=img_shape[1]:
       #             print("y",y_pt)
                    y_pt = img_shape[1]-1
        #        print(x_pt, y_pt)
                lp_curr[i][j] = input_img[x_pt][y_pt]
                j += 1
            i += 1

		# Applying Gaussian blur
        lp_curr = skimage.filters.gaussian(lp_curr, sigma=1.)
        large_patch.append(lp_curr)

		# Obtaining small patch with spacing of 5 pixels
        for x in range(0, 40, 5):
            for y in range(0, 40, 5):
                x_new = int(x/5)
                y_new = int(y/5)
                sp_curr[x_new][y_new] = lp_curr[x][y]

		# Applying Bias and gain normalization: This step makes our features invariant to overall intensity differences and RGB distribution in image
        sp_curr = (sp_curr - sp_curr.mean())/sp_curr.std()

        features.append(sp_curr)
        harris_pt.append(pt)

    """
	# Display first five feature descriptors
    for i in range(5):
        fp, h_pt = features[i], harris_pt[i]
        implot = plt.imshow(color_img)
        plt.scatter(x=h_pt[0], y=h_pt[1], c='r', s=40)
        plt.show()
        sk.imshow(fp)
        sk.show()
    """
    return features, harris_pt


def feature_match(fd1, fd2, pt1, pt2):

    """
	Implement section 5 of obtaining matching features as mentioned in "Multi-Image Matching using Multi-Scale Oriented Patches" by Brown et. al
	"""
    
    feature_desc, match_pt, img_pt1, img_pt2 = [], [], [], []
    fd1_len = len(fd1)
    fd2_len = len(fd2)
    
    # Iterating through each feature patch to get best and second best match
    for i in range(fd1_len):
        feature1 = fd1[i]
        img1_pt = pt1[i]

		# Variables to track this feature patch's best/second best
        best_match_fd2 = None
        best_match_error = float("inf")
        best_match_pt = None
        second_best_match_fd2 = None
        second_best_match_error = float("inf")
        second_best_match_pt = None

        for j in range(fd2_len):
            feature2 = fd2[j]
            img2_pt = pt2[j]

			# Calculating error between patches using Sum of Squared Differences(SSD)
            error = calculate_error(feature1, feature2)
			
			# Condition to check matching feature patches
            if best_match_error > error:
                second_best_match_error = best_match_error
                second_best_match_fd2 = best_match_fd2
                second_best_match_pt = best_match_pt
                best_match_fd2 = feature2
                best_match_error = error
                best_match_pt = img2_pt

            elif second_best_match_error > error:
                second_best_match_fd2 = feature2
                second_best_match_error = error
                second_best_match_pt = best_match_pt

        # As mentioned in paper the ratio of 1-NN (best match) to 2-NN (second best match) is better metric to evaluate correct matches
        ratio = float(best_match_error/(second_best_match_error))

        if ratio < 0.4:
            match_pt.append([img1_pt, best_match_pt])
            img_pt1.append(img1_pt)
            img_pt2.append(best_match_pt)
            feature_desc.append([best_match_fd2, second_best_match_fd2])

    return match_pt, feature_desc, img_pt1, img_pt2

def calculate_error(point1, point2):
    # calculate error between patches using Sum of Squared Differences(SSD)
    dist = np.sum((point1-point2)**2)
    return dist

def ransac_calculate(matches, img1, img2, k=5000):
	
    """
	Returns set of points to do Homography on after removing outliers
	"""
    
    H = np.eye(3)
    inliers = []
    
	# Loop large number of times
    for i in range(0, k):
		# Randomly select 4 corresponding points (total 8)
        pts1, pts2 = get_Random(matches)
        #print("mat",pts1)
        curr_H = Homography_matrix(pts1, pts2) 
        
        O_corners = get_corners(img1)
        T_corners = corner_transformation(O_corners, curr_H)
        # print(len(T_corners),curr_H.shape)
        T = translation(T_corners)
        T_corners_translated = corner_transformation(O_corners, np.linalg.inv(T.dot(curr_H)))
        
        cur_inliers = []

		# Iterate through remaining points
        for pt in matches:
            p1, p2 = pt
            if p1 not in pts1 and p2 not in pts2:
                p2_transformed = corner_transformation([p2], np.dot(curr_H, T))[0]
                x1, y1 = p1
                x2, y2 = (p2_transformed[0], p2_transformed[1])
                error = math.sqrt((x2-x1)**2 + (y2-y1)**2)
				
                if error < 20:
                    cur_inliers.append(pt)

		# correct homography
        if len(cur_inliers) > len(inliers):
            inliers = cur_inliers
            H = curr_H

    return H,inliers

def Homography_matrix(pts_1, pts_2):
	
    """
	Returns matrix that will transform pts_1 to pts_2
	"""
	
    mat1, mat2 = [], []
    for i in range(4):
        x_1, y_1 = pts_1[i]
        x_2, y_2 = pts_2[i]
        row1 = np.asarray([x_1, y_1, 1., 0, 0, 0, -x_1*x_2, -y_1*x_2])
        row2 = np.asarray([0, 0, 0, x_1, y_1, 1., -x_1*y_2, -y_1*y_2])
        mat1.append(row1)
        mat1.append(row2)
        mat2.append([x_2])
        mat2.append([y_2])

    mat1 = np.asarray(mat1)
    mat2 = np.asarray(mat2)

    # Least squares solution for matrices
    h = np.linalg.lstsq(mat1, mat2)[0]
    H = []
    for e in h:
        H.append(e[0])
    H.append(1.)
    H = np.asarray(H).reshape((3,3))
    return H

def get_corners(img):
	h, w, c = img.shape
	ltc = (float(0), float(0))
	lbc = (float(w), float(0))
	rtc = (float(0), float(h))
	rbc = (float(w), float(h))
	return [ltc, lbc, rtc, rbc]

def corner_transformation(pts, H):
	
	corner_pts = []
	for pt in pts:
		pt = list(pt)
		pt.append(1.)
		corner_pts.append(pt)
       
	return corner_pts#np.dot(corner_pts, H)
    
def translation(transformed):
	temp = []
	for corner in transformed:
		x, y, z = corner
        # Dividing values by z as we are expecting the points to be on z-plane
		temp.append((x/z, y/z))

	xs = [pt[0] for pt in temp]
	ys = [pt[1] for pt in temp]

	# Calculate translation
	A = np.identity(3)
	offset_x = -min(xs)
	offset_y = -min(ys)
	A[0][2] = offset_x
	A[1][2] = offset_y
	return A

def get_Random(points):
    pt1,pt2=[],[]
    for n in range(4):
        pt = random.choice(points)
        pt1.append(pt[0])
        pt2.append(pt[1])
    return pt1,pt2

def warping(image1, image2, H_mat1, H_mat2):
	
    """
	This function is responsible to warp the first image to second, and change the size of both images
	"""
    
    h, w, c = img1.shape
    new_dim = (h, w+image2.shape[1], c)

    d = dict()
	
    O_corners = get_corners(image1)
    T_corners = corner_transformation(O_corners, H_mat1)
    T = translation(T_corners)

    T_corner_translate = corner_transformation(O_corners, np.linalg.inv(T.dot(H_mat1)))
    c_x = [int(c[0]) for c in list(T_corner_translate)]
    c_y = [int(c[1]) for c in list(T_corner_translate)]
    new_w = max(max(c_x), w)
    new_h = max(max(c_y), h)
    new_dim = (new_h, new_w, 3)

    im_1 = skimage.transform.warp(image1, np.linalg.inv(T.dot(H_mat1)), output_shape=new_dim)
    im_2 = skimage.transform.warp(image2, H_mat2, output_shape=new_dim)
    
    return im_1, im_2,T.dot(H_mat1),T_corner_translate

def mosaic_create(img1_warp, img2_warp, img1, img2, T_corners_translated):
	h, w, c = img1.shape
	new_dim = img1_warp.shape
	base = np.zeros(new_dim)

	#Calculate center of image1,image2 and complete image
	center1 = [w/2.,h/2.]
	center2 = [sum([pt[0] for pt in T_corners_translated])/4., sum([pt[1] for pt in T_corners_translated])/4.]
	center = ((center1[0]+center2[0])/2,(center1[1]+center2[1])/2)

	for r in range(int(new_dim[0])):
		for c in range(int(new_dim[1])):
            # Condition if there are black pixels in second image
			if img2_warp[r][c][0] == 0 and img2_warp[r][c][1] == 0 and img2_warp[r][c][2] == 0:
				base[r][c] = img1_warp[r][c]
            # Condition if there are black pixels in first image
			elif img1_warp[r][c][0] == 0 and img1_warp[r][c][1] == 0 and img1_warp[r][c][2] == 0:
				base[r][c] = img2_warp[r][c]
			else:
				numer = float((c-center1[0]))
				denom = (center2[0]-center1[0])
				if denom != 0:
					alpha = float(numer/denom)
					base[r][c] = img1_warp[r][c] * alpha + (1-alpha)*img2_warp[r][c]
				else:
					base[r][c] = img1_warp[r][c]

	return base

import sys
class BandBlend:
    def __init__(self):
        self.levels=0
    
    def preprocess_images(self,img1, img2, width, half):
        
        """
        This function preprocesses images to generate mask for further process
        """
        
        if img1.shape[0] != img2.shape[0]:
            print ("error: image dimension error")
            sys.exit()
        if width > img1.shape[1] or width > img2.shape[1]:
            print ("error: overlapped area too large")
            sys.exit()
    
        w1 = img1.shape[1]
        w2 = img2.shape[1]
    
        if half:    # Creating mask with half black and other half as white
            new_dim = np.array(img1.shape)
            new_dim[1] = w1/2 + w2/2    # Width of new image
    
            image1 = np.zeros(new_dim)
            image1[:, :int(w1/2 + width/2)] = img1[:, :int(w1/2 + width/2)]
            image2 = np.zeros(new_dim)
            image2[:, int(w1/2 - width/2):] = img2[:,int(w2 - (w2/2 + width/2)):]
            mask = np.zeros(new_dim)
            mask[:, :int(w1/2)] = 1
        else:
            new_dim = np.array(img1.shape)
            new_dim[1] = w1 + w2 - width
    
            image1 = np.zeros(new_dim)
            image1[:, :w1] = img1
            image2 = np.zeros(new_dim)
            image2[:, int(w1 - width):] = img2
            mask = np.zeros(new_dim)
            mask[:, :int(w1 - width/2)] = 1
    
        return image1,image2,mask
    
    def calculate_levels(self,image1,image2):
        
        """
        Calculating number of levels involved in Laplacian and Gaussian Pyramid Blending
        """
        
        self.levels = int(np.floor(np.log2(min(image1.shape[0], image1.shape[1],image2.shape[0], image2.shape[1]))))
        
        
    def Gaussian_Blend(self,image):
        
        """
        Applying the method to create Gaussian Pyramid 
        """
        gaussian_arr=[image]
        
        for i in range(self.levels-1):
            res=cv2.pyrDown(gaussian_arr[i])
            gaussian_arr.append(res)
            #cv2.imshow("rew",res)
            #cv2.waitKey(0)
        
        return gaussian_arr
        
    def Laplacian_Blend(self,image):
        
        """
        Applying the method to create Laplacian Pyramid 
        """
        laplacian_arr=[]
        
        for i in range(self.levels - 1):
            next_img = cv2.pyrDown(image)
            # print(i,image.shape,next_img.shape,cv2.pyrUp(next_img, image.shape[1::-1]).shape)
            pyr_up = cv2.pyrUp(next_img, image.shape[1::-1])
            if image.shape != pyr_up.shape:
                image = cv2.resize(image,(pyr_up.shape[1],pyr_up.shape[0]))
            # print(image.shape,pyr_up.shape)
            res = image - pyr_up
            laplacian_arr.append(res)
            #cv2.imshow("rew",res)
            #cv2.waitKey(0)
            image = next_img
        laplacian_arr.append(image)
        return laplacian_arr
    
    def blend_images(self,Laplacian_img1, Laplacian_img2, Gaussian_mask):
        
        """
        Implementing blending of the two images using gaussian mask by the following formula:
            LS(i,j) = GR(I,j,)*LA(I,j) + (1-GR(I,j))*LB(I,j)
        where, LA and LB are laplacian pyramids of img1 and img2 and GR is the gaussian pyramid of selected region
        """
        blended_res=[]
        # print(len(Laplacian_img1),len(Gaussian_mask))
        for i,GR in enumerate(Gaussian_mask):
            # print(Laplacian_img1[i].shape,Laplacian_img2[i].shape,GR.shape)
            GR = cv2.resize(GR,(Laplacian_img1[i].shape[1],Laplacian_img1[i].shape[0]))
            res = GR*Laplacian_img1[i] + (1-GR)*Laplacian_img2[i]
            blended_res.append(res)
            #cv2.imshow("rew",res)
            #cv2.waitKey(0)
        
        return blended_res
    
    def create_final_mosaic(self,b_img):
        
        """
        Reconstruction of blended image to generate final stitched image
        """
        b = b_img[-1]
        # print(b.shape)
        for l in b_img[-2::-1]:
            # print(b.shape)
            b = cv2.pyrUp(b, l.shape[1::-1])
            if b.shape != l.shape:
                l = cv2.resize(l,(b.shape[1],b.shape[0]))
            # print(b.shape,l.shape)
            b += l
        
        return b
    
    def multi_band_mosiac(self,image1,image2,width,flag,c,mask1):
        if width < 0:
            print ("error: width to be overlapped should be a positive integer")
            sys.exit()
        self.calculate_levels(image1, image2)
        processed1, processed2, mask = self.preprocess_images(image1, image2, width, flag)
        # print(self.levels)
        
        # cv2.imshow("c1",mask)
        # cv2.waitKey(0)
        """
        cv2.imshow("c2",processed2)
        cv2.waitKey(0)
        """
        # Get Gaussian pyramid and Laplacian pyramid
        GR = self.Gaussian_Blend(mask)
        LPA =self.Laplacian_Blend(processed1)
        LPB = self.Laplacian_Blend(processed2)
        
        
        
        # Blend two Laplacian pyramidspass
        blended = self.blend_images(LPA, LPB, GR)
        #cv2.imshow("b",blended)
        #cv2.waitKey(0)
    
        # Reconstruction process
        final_img = self.create_final_mosaic(blended)
        cv2.imshow("re",final_img)
        cv2.waitKey(0)
        final_img[final_img > 255] = 255
        final_img[final_img < 0] = 0
        implot = plt.imshow(img1)
    
        return final_img
            

img1 = cv2.imread("/Users/nimishamittal/Documents/OldProjects/ML/Image_Color/Morphing/im1_l.jpg")
img2 = cv2.imread("/Users/nimishamittal/Documents/OldProjects/ML/Image_Color/Morphing/im1_r.jpg")

# print(img1.shape,img2.shape)
grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
input_arr = [[img1,grayImg1],[img2,grayImg2]]
harris_points(input_arr)
"""
h, curr_coords = harris.get_harris_corners(grayImg1)
print(h.shape)
print("curr_coords",curr_coords.shape)
# Red dots of size 40
plt.scatter(x=curr_coords[1], y=curr_coords[0], c='r', s=10)
plt.show()


h1, curr_coords1 = harris.get_harris_corners(grayImg2)

#print("curr_coords",curr_coords.shape)
implot = plt.imshow(img2)
# Red dots of size 40
plt.scatter(x=curr_coords1[1], y=curr_coords1[0], c='r', s=10)
plt.show()
"""