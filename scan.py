# USAGE
# python scan.py --image images/page.jpg 

# import the necessary packages
from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[1] / 500.0
orig = image.copy()
image = imutils.resize(image, width = 500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print "STEP 1: Edge Detection"
# cv2.imshow("Image", image)
# cv2.imshow("Edged", edged)
cv2.imwrite('curImg.png', image)
cv2.imwrite('edgesImg.png', edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
_, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:4]

lastApprox = [0,0]
danApproxes = []
# loop over the contours
for c in cnts:
	print cv2.contourArea(c)
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		# if approx[0] # todo calc if within 5% either direction and discard if so
		screenCnt = approx
		danApproxes.append(approx)
		break
		# continue

# show the contour (outline) of the piece of paper
print "STEP 2: Find contours of paper"
cv2.drawContours(image, danApproxes, -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
cv2.imwrite('outlineImg.png',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped, 251, offset = 10)
warped = warped.astype("uint8") * 255

# show the original and scanned images
print "STEP 3: Apply perspective transform"
cv2.imwrite('origResized.png',imutils.resize(orig, width=650))
# cv2.imwrite('scannedResized.png',imutils.resize(warped, height=650))
cv2.imwrite('scannedResized2.png',warped)
from PIL import Image
im = Image.fromarray(warped).crop((215, 416, 950, 481))
im.save('scannedRS3.jpg')
# cv2.imshow("Original", imutils.resize(orig, height = 650))
# cv2.imshow("Scanned", imutils.resize(warped, height = 650))
# cv2.waitKey(0)