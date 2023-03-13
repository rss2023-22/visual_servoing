import cv2
import numpy as np
import pdb

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""

	def lineFollowingImgCreation(img):
		lowBound = 200
		upBound = 300
		dims = img.shape
		for i in range(dims[0]):
			for j in range(dims[1]):
				if i<lowBound or i>upBound:
					img[i][j] = (1,1,1)
		return img


	def lookupBounds(x):
		if x == 1:	# decent mask, noise
			lower_bound = np.array([5,80,80])
			upper_bound = np.array([20,255,255])
		elif x == 2: # better mask, less noise, not sensitive to lighter orange
			lower_bound = np.array([10,100,100])
			upper_bound = np.array([16,255,255])
		elif x == 3: # similar to 2, VERY low noise (might be functional)
			lower_bound = np.array([10,150,150])
			upper_bound = np.array([16,255,255])
		elif x == 4: # similar to 3 but more agressive on eliminating noise
			lower_bound = np.array([5,150,200])
			upper_bound = np.array([20,255,255])
		elif x == 5: # similar to 4 but more sensitive to lighter oranges, prob best alone
			lower_bound = np.array([5,180,190])
			upper_bound = np.array([25,255,255])
		elif x == 6: # similar to 5 but more sensitive to lighter oranges, probably the best so far with erode and dilate. Good with 1 and 2 iters. 
			lower_bound = np.array([5,180,190]) # 180 on the last!
			upper_bound = np.array([35,255,255])
		elif x == 7: # similar to 5 but more sensitive to lighter oranges, probably the best so far with erode and dilate. Good with 2 and 5 iters. 
			lower_bound = np.array([5,180,190])
			upper_bound = np.array([35,255,255])
		elif x == 9: # FOR TAPE!!
			lower_bound = np.array([1,100,50])
			upper_bound = np.array([35,255,255])
		return [lower_bound,upper_bound]

	# PARAMS
	########
	# decent combos: bounds=6, 1 iter, 2 iter || bounds=7, 2 iter, 6 iter
	viz_original_img = False
	viz_masked_img = False
	viz_eroded = False
	viz_dilated = False
	viz_box = False
	set_bounds = 7 # 1,2,3,4,5,6
	line_following = True

	# BEGIN CODE 
	############

	# step 0: limit range if line following
	if line_following:
		img = lineFollowingImgCreation(img)
		set_bounds = 9
	if viz_original_img:
		image_print(img) # see original image


	# step 1: convert to HSV color scheme (more robust to changes in illumination... img shape is like [[[H,S,V]...]...]
	image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# step 2: use cv2.inRange to bound the HSV values, only keeping what is useful (orange). Perhaps improve ranges. 
	bounds = lookupBounds(set_bounds)
	imagemask = cv2.inRange(image,bounds[0],bounds[1])
	if viz_masked_img:
		image_print(imagemask)

	# step 3: use erosion and dilution
	kernel1 = np.ones((5,5), np.uint8)
	image_erod = cv2.erode(imagemask,kernel1,iterations=2) # 1 # 2
	if viz_eroded:
		image_print(image_erod)
	image_dila = cv2.dilate(image_erod,kernel1,iterations=5) # 2 # 6
	if viz_dilated:
		image_print(image_dila)

	# step 4: get contours
	ret,thresh = cv2.threshold(image_dila,127,255,0)
	_, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # CHANGE BCK
	if len(contours) == 0: # no box:
		return ((0,0),(0,0))
	
	if line_following:
		cnt = contours[0]
	else:
		cnt = contours[-1]

	# step 5: get bounding box
	x,y,w,h = cv2.boundingRect(cnt)
	bounding_box = ((x,y),(x+w,y+h))
	print(bounding_box)

	# step 6: display original img with bounding rectangle!
	img = cv2.rectangle(img,bounding_box[0],bounding_box[1],(0,255,0),2)
	if viz_box:
		image_print(img)

	# Return bounding box
	return bounding_box
