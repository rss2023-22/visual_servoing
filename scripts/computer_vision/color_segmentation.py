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
			return img[150:275], 150, 275, None, None


	def lookupBounds(line_following,color):
		if not line_following: # for following orange cone 
			lower_bound = np.array([5,180,190])
			upper_bound = np.array([35,255,255])
		else: # for line following orange tape
			if color == 'orange':
				lower_bound = np.array([1,100,50]) #np.array([1,100,50]) # upper_bound = np.array([35,255,255])
				upper_bound = np.array([35,255,255])
			if color == 'white':
				lower_bound = np.array([10,0,130]) #np.array([1,100,50]) # upper_bound = np.array([35,255,255])
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
		viz_dilated = True
		viz_edge = False
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
			kernel1 = np.ones((2,2), np.uint8)
		image_erod = cv2.erode(imagemask,kernel1,iterations=2) # 2 orange line
		if testing and viz_eroded:
			image_print(image_erod)
		image_dila = cv2.dilate(image_erod,kernel1,iterations=5) # 5 orange line # 
		if testing and viz_dilated:
			image_print(image_dila)
		image_edge = cv2.Canny(image_dila,50,200,None,3)
		if testing and viz_edge:
			image_print(image_edge)
			
	lines = cv2.HoughLines(image_edge,1,np.pi/180,45,None,0,0)
	baseline = len(imgOrig)+300
	eval_line = lambda line,y: (line[0][0]-np.sin(line[0][1])*(y-lowBoundvert))/np.cos(line[0][1])
	best_left,best_right = None,None
	if lines is not None:
		for i in range(0,len(lines)):
			if abs(np.cos(lines[i][0][1])) < 0.01: continue
			e = eval_line(lines[i],baseline)
			if e < len(imgOrig[0])*0.5 and (best_left is None or e > eval_line(best_left,baseline)):
				best_left = lines[i]
			elif e >= len(imgOrig[0])*0.5 and (best_right is None or e < eval_line(best_right,baseline)):
				best_right = lines[i]
			#imgOrig = cv2.line(imgOrig,(int(rho/a),lowBoundvert),(int((rho-b*(len(imgOrig)-lowBoundvert))/a),len(imgOrig)),(255,128,0),2)
		#if testing and viz_box: image_print(imgOrig)
	
	if best_left is None or best_right is None:
		#TODO: some kind of standardized way to say that not enough lines are detected
		return ((0,0),(1,1))
	start_x,start_y = (int(eval_line(best_left,lowBoundvert)+eval_line(best_right,lowBoundvert))//2,lowBoundvert) 
	end_x,end_y = (int(eval_line(best_left,len(imgOrig))+eval_line(best_right,len(imgOrig)))//2,len(imgOrig))
	slope = (end_y - start_y)/float((end_x-start_x)+0.001)

	
	midpoint_x,midpoint_y = (start_x+end_x)/2 , (start_y+end_y)/2
	bb_size = 10
	constant_offset_y = -90
	constant_offset_x = constant_offset_y/slope
	bb = ((int(midpoint_x-bb_size+constant_offset_x),int(midpoint_y-bb_size+constant_offset_y)),(int(midpoint_x+bb_size+constant_offset_x),int(midpoint_y+bb_size+constant_offset_y)))
	#print(bb)
	if testing and viz_box:
		imgOrig = cv2.line(imgOrig,(int(eval_line(best_left,lowBoundvert)),lowBoundvert),(int(eval_line(best_left,len(imgOrig))),len(imgOrig)),(255,128,0),2)
		imgOrig = cv2.line(imgOrig,(int(eval_line(best_right,lowBoundvert)),lowBoundvert),(int(eval_line(best_right,len(imgOrig))),len(imgOrig)),(255,128,0),2)
		imgOrig = cv2.line(imgOrig,(start_x,start_y),(end_x,end_y),(0,255,0),3)
		img = cv2.rectangle(imgOrig,bb[0],bb[1],(0,255,0),2)
		image_print(img)
		
	return bb