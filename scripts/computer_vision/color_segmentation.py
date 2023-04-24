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

def cd_color_segmentation(img, template = None, line_following = 1.0, testing = True,lowBound = 225,upBound=275):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		params: (lineFollowing,Testing); (int,Bool)
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""

	def cropImg(img,line_color):
		if line_color == 'orange':
			img = img[225:275]
			return img,225,275, None, None
		if line_color == 'white':
			return img[150:275,0:400], 150, 275, 0, 400


	def lookupBounds(line_following,color):
		if not line_following: # for following orange cone 
			lower_bound = np.array([5,180,190])
			upper_bound = np.array([35,255,255])
		else: # for line following orange tape
			if color == 'orange':
				lower_bound = np.array([1,100,50]) #np.array([1,100,50]) # upper_bound = np.array([35,255,255])
				upper_bound = np.array([35,255,255])
			if color == 'white':
				lower_bound = np.array([20,0,160]) #np.array([1,100,50]) # upper_bound = np.array([35,255,255])
				upper_bound = np.array([190,50,255])
		return [lower_bound,upper_bound]

	# SET PARAMS
	if line_following == 1.0:
		line_following = True
		# line_color = 'orange'
		line_color = 'white'

	else:
		line_following = False
	if testing:
		testing = True
		viz_original_img = False
		viz_masked_img = False
		viz_eroded = False
		viz_dilated = False
		viz_box = True
	else:
		testing = False

	# BEGIN CODE 
	############
	imgOrig = img

	# step 0: limit range if line following
	if line_following: # crop image
		img, lowBoundvert, upBoundvert, lowboundSide, upboundSide = cropImg(img,line_color)

	if testing and viz_original_img:
		image_print(img) # see original image


	# step 1: convert to HSV color scheme (more robust to changes in illumination... img shape is like [[[H,S,V]...]...]
	image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# step 2: use cv2.inRange to bound the HSV values, only keeping what is useful (orange). Perhaps improve ranges. 
	bounds = lookupBounds(line_following,line_color)
	imagemask = cv2.inRange(image,bounds[0],bounds[1])
	if testing and viz_masked_img:
		image_print(imagemask)

	# step 3: use erosion and dilution
	kernel1 = np.ones((5,5), np.uint8)
	if not line_following:
		image_erod = cv2.erode(imagemask,kernel1,iterations=1) # 1 cone 
		if testing and viz_eroded:
			image_print(image_erod)
		image_dila = cv2.dilate(image_erod,kernel1,iterations=2) # 2 cone 
		if testing and viz_dilated:
			image_print(image_dila)
	else:
		if line_color == 'white':
			kernel1 = np.ones((4,4), np.uint8)
		image_erod = cv2.erode(imagemask,kernel1,iterations=2) # 2 orange line
		if testing and viz_eroded:
			image_print(image_erod)
		image_dila = cv2.dilate(image_erod,kernel1,iterations=5) # 5 orange line # 
		if testing and viz_dilated:
			image_print(image_dila)


	# step 4: get contours, if multiple, take contour closest to the center of img (for line following)
	ret,thresh = cv2.threshold(image_dila,127,255,0)
	if testing:
		contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # opencvNew version
	else:
		_, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # opencvOld version
	if len(contours) == 0: # no box!
		return ((0,0),(0,0))
	
	def getDist(coords1,coords2): # x coord only
		return abs(coords1[0]-coords2[0])
	
	if line_following:
		if line_color == 'orange':
			image_center = np.asarray(image_dila.shape) / 2
			image_center = tuple(image_center.astype('int32'))
			closest_contour = None 
			min_dist = float('Inf')
			for contour in contours:
				M = cv2.moments(contour)
				center_X = int(M["m10"] / M["m00"]); center_Y = int(M["m01"] / M["m00"])
				distances_to_center = getDist((image_center[1],image_center[0]), (center_X,center_Y))
				if distances_to_center < min_dist:
					min_dist = distances_to_center; closest_contour = contour
			cnt = closest_contour
		if line_color == 'white':
			image_center = np.asarray(image_dila.shape) / 2
			image_center = tuple(image_center.astype('int32'))
			closest_contour = None 
			min_dist = float('Inf')
			for contour in contours:
				M = cv2.moments(contour)
				center_X = int(M["m10"] / M["m00"]); center_Y = int(M["m01"] / M["m00"])
				if center_X < min_dist:
					min_dist = center_X
					closest_contour = contour
				# print((center_X,center_Y))
				# if center_X > 300:
				# 	continue
				# distances_to_center = getDist((image_center[1],image_center[0]), (center_X,center_Y))
				# if distances_to_center < min_dist:
				# 	min_dist = distances_to_center; closest_contour = contour
			cnt = closest_contour
	else:
		cnt = contours[-1]

	# step 5: get bounding box
	x,y,w,h = cv2.boundingRect(cnt)
	if line_following:
		if upboundSide != None:
			constant_offset = 200
			bounding_box = ((x+lowboundSide+constant_offset,y+lowBoundvert),(x+w+constant_offset,y+upBoundvert))
		else:
			bounding_box = ((x,y+lowBoundvert),(x+w,y+h+upBoundvert))
	else:
		bounding_box = ((x,y),(x+w,y+h))

	# step 6: display original img with bounding rectangle!
	if testing:
		img = cv2.rectangle(imgOrig,bounding_box[0],bounding_box[1],(0,255,0),2)
		if viz_box:
			image_print(img)

	return bounding_box
