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

def cd_color_segmentation(img, template = None, line_following = 1.0, testing = True,lowBound = 220,upBound=376): # 250
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		params: (lineFollowing,Testing); (int,Bool)
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""

	def cropImg(img,lowBound,upBound):
		img = img[lowBound:upBound]
		return img,lowBound,upBound


	def lookupBounds(line_following):
		if not line_following: # for following orange cone 
			lower_bound = np.array([5,180,190])
			upper_bound = np.array([35,255,255])
		else: # for line following orange tape
			lower_bound = np.array([0,120,70]) #np.array([1,100,50]) # upper_bound = np.array([35,255,255])
			upper_bound = np.array([50,255,255])
		return [lower_bound,upper_bound]

	# SET PARAMS
	if line_following == 1.0:
		line_following = True
	else:
		line_following = False
	if testing:
		testing = True
		viz_original_img = True
		viz_masked_img = True
		viz_eroded = True
		viz_dilated = True
		viz_box = True
	else:
		testing = False

	# BEGIN CODE 
	############
	imgOrig = img
	_, width, _ = np.shape(img)

	# step 0: limit range if line following
	if line_following: # crop image
		img, lowBound, upBound = cropImg(img,lowBound,upBound)

	if testing and viz_original_img:
		image_print(img) # see original image


	# step 1: convert to HSV color scheme (more robust to changes in illumination... img shape is like [[[H,S,V]...]...]
	image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# step 2: use cv2.inRange to bound the HSV values, only keeping what is useful (orange). Perhaps improve ranges. 
	bounds = lookupBounds(line_following)
	imagemask = cv2.inRange(image,bounds[0],bounds[1])
	if testing and viz_masked_img:
		image_print(imagemask)

	# step 3: use erosion and dilution
	kernel1 = np.ones((8,8), np.uint8)
	kernel2 = np.ones((10,10), np.uint8)
	#kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (21,2))
	if not line_following:
		image_erod = cv2.erode(imagemask,kernel1,iterations=1) # 1 # 2
		if testing and viz_eroded:
			image_print(image_erod)
		image_dila = cv2.dilate(image_erod,kernel1,iterations=2) # 2 # 5
		if testing and viz_dilated:
			image_print(image_dila)
	else:
		image_erod = cv2.erode(imagemask,kernel1,iterations=1)
		if testing and viz_eroded:
			image_print(image_erod)
		image_dila = cv2.dilate(image_erod,kernel2,iterations=3)
		if testing and viz_dilated:
			image_print(image_dila)



	# step 4: get contours, if multiple, take contour closest to the center of img (for line following)
	ret,thresh = cv2.threshold(image_dila,127,255,0)
	contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # opencvNew version
	if len(contours) == 0: # no box!
		return ((0,0),(0,0))
	
	def getDist(coords1,coords2): # x coord only
		return abs(coords1[0]-coords2[0])
	
	if line_following:
		image_center = np.asarray(image_dila.shape) / 2
		image_center = tuple(image_center.astype('int32'))
		closest_contour = [0,0] 
		min_dist = float('-Inf')
		for contour in contours:
			M = cv2.moments(contour)
			center_X = int(M["m10"] / M["m00"]); center_Y = int(M["m01"] / M["m00"])
			#distances_to_center = getDist((image_center[1],image_center[0]), (center_X,center_Y))
			x,y,w,h = cv2.boundingRect(contour)
			area = cv2.contourArea(contour)
			if area > min_dist:
				min_dist = area; closest_contour = contour
		cnt = closest_contour
	else:
		cnt = contours[-1]

	
	if len(cnt) < 3 or cv2.contourArea(cnt) < 2000:
		return ((0,0),(0,0))

	# area = cv2.contourArea(cnt)
	# if area < 5000:
	# 	return ((-1,-1),(-1,-1))

	vx,vy,x,y = cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
	lefty = int((-x*vy/vx)+y)
	righty = int(((width-x)*vy/vx)+y)
	#print(line)

	# step 5: get bounding box
	x,y,w,h = cv2.boundingRect(cnt)
	#print(image_center)


	if line_following:
		bounding_box = ((x,y+lowBound),(x+w,y+h+lowBound))
		middle = 336
		x1,x2 = bounding_box[0][0], bounding_box[1][0]
		if abs(x1-middle) > 240 or abs(x2-middle) > 240:
			if vy/vx > 0 and x1 < 100: 
				bb = ((x1-5,min(y+lowBound-5,y+lowBound+5)),(x1+5,max(y+lowBound-5,y+lowBound+5)))
			else:
				bb = ((x2-5,y+lowBound-5),(x2+5,y+lowBound+5))
		else:
			bb = bounding_box

		bounding_box = bb
		#print(bounding_box)
	else:
		bounding_box = ((x,y),(x+w,y+h))

	# step 6: display original img with bounding rectangle!
	if testing:
		img = cv2.rectangle(imgOrig,bounding_box[0],bounding_box[1],(0,255,0),2)
		if viz_box:
			image_print(img)
	

	return bounding_box
