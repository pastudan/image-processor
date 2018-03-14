# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
from __future__ import division
from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from math import sqrt
from PIL import Image, ImageOps, ImageEnhance
import argparse
import cv2
import numpy
import io
import re
import json

DISTANCE_THRESHOLD = 20
AREA_THRESHOLD = 0.5

DESIRED_COMBINED_CODE_WIDTH = 400

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[1] / 500.0
orig = image.copy()
image = imutils.resize(image, width=500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# save the original image and the edge detected image
# print"STEP 1: Edge Detection"
cv2.imwrite('output-images/1-original.png', orig)
cv2.imwrite('output-images/1-edges.png', edged)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
_, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)


def distance(pointA, pointB):
    dX = pointA[0][0] - pointB[0][0]
    dY = pointA[0][1] - pointB[0][1]
    return sqrt(dX ** 2 + dY ** 2)


lastContourArea = 1
gamePieceGeometries = []
duplicatePieceGeometries = []

# loop over the contours
for contour in contours:
    area = cv2.contourArea(contour)

    # when there's a dramatic shift in contour surface areas (i.e. 1/2 as small as the last one),
    # then we've probably stopped finding game pieces
    if area / lastContourArea < AREA_THRESHOLD:
        break

    # approximate the contour
    perimeter = cv2.arcLength(contour, True)
    newGeometry = cv2.approxPolyDP(contour, 0.05 * perimeter, True)

    # if our approximated contour does not have four points,
    # then it's probably not a game piece and we can discard it
    print len(newGeometry)
    if len(newGeometry) != 4:
        break

    lastContourArea = area

    # print"%s, %s, %s, %s" % (newGeometry[0][0], newGeometry[1][0], newGeometry[2][0], newGeometry[3][0])

    # sometimes OpenCV finds multiple similar & overlapping shape geometries,
    # so lets make sure this geometry doesn't already exist in our list of found game pieces
    for existingGeometry in gamePieceGeometries:
        # assign existing piece geometry to a new variable
        # since we will be removing points once found
        existingPoints = existingGeometry
        similarPoints = 0
        for newPoint in newGeometry:
            for index, existingPoint in enumerate(existingPoints):
                if distance(newPoint, existingPoint) <= DISTANCE_THRESHOLD:
                    similarPoints += 1
                    # This point is similar to one of the existing ones.
                    # Delete it so we don't try to match it to a new one next time.
                    numpy.delete(existingPoints, index)
                    break
        if similarPoints == 4:
            # all four points in this new geometry are similar to an
            # existing piece geometry, so this piece is not unique.
            # print"similar rectangle found!"
            duplicatePieceGeometries.append(newGeometry)
            break

    else:
        # if we don't overlap with a similar rectangle that we've found already
        # (ie we didn't `break` above), then add this piece to the list of found geometries
        firstPieceContour = newGeometry
        gamePieceGeometries.append(newGeometry)

# print"%s game pieces found" % len(gamePieceGeometries)

# save the contour (outline) of the piece of paper
# print"STEP 2: Find contours of paper"
cv2.drawContours(image, gamePieceGeometries, -1, (0, 255, 0), 2)
# cv2.drawContours(image, duplicatePieceGeometries, -1, (0, 0, 255), 2)
cv2.imwrite('output-images/2-contours.png', image)

# these aspect ratios come from directly measuring the monopoly game pieces
A = 0.306122449  # aspect ratio of fully stripped pieces
B = 0.394557823  # aspect ratio of single-stripped pieces
C = 0.472972973  # aspect ratio of in tact pieces
smallAspectRatioThreshold = (B - A) / 2 + A  # = 0.350340136  #small (fully stripped) piece threshold
largeAspectRatioThreshold = (C - B) / 2 + B  # = 0.433765398  #large (in tact) piece threshold
singleStripDelta = (B - A) * 2  # = 0.088435374  #  proportional height of one strip on a single-strip piece
multiStripDelta = (C - A) / 2  # = 0.083425262  #  proportional height of each strip on a multi-strip piece


def get_average_brightness(brightness_image):
    r, g, b = 0, 0, 0
    count = 0
    brightness_image = brightness_image.convert('RGB')
    width, height = brightness_image.size
    for s in range(0, width):
        for t in range(0, height):
            pixlr, pixlg, pixlb = brightness_image.getpixel((s, t))
            r += pixlr
            g += pixlg
            b += pixlb
            count += 1
    return ((r / count) + (g / count) + (b / count)) / 3


onlineCodeImages = []
manualCodeImages = []

for index, geometry in enumerate(gamePieceGeometries):
    # print"########### %s ###########" % index
    # apply the four point transform to obtain a
    # top-down view of the original image
    warped = four_point_transform(orig, geometry.reshape(4, 2) * ratio)

    # STEP 3: Apply perspective transform
    cv2.imwrite("output-images/3-warped-%s.png" % index, warped)

    # STEP 4: Crop only desired areas
    pilImage = Image.fromarray(warped)
    width, height = pilImage.size

    aspectRatio = height / width
    if smallAspectRatioThreshold < aspectRatio < largeAspectRatioThreshold:
        stripHeight = singleStripDelta * height * 1.3

        topStrip = pilImage.crop((0, 0.1 * stripHeight, width, stripHeight))
        bottomStrip = pilImage.crop((0, height - stripHeight, width, height - 0.1 * stripHeight))

        # topStrip.save("output-images/4-topStrip-%s.png" % index)
        # bottomStrip.save("output-images/4-bottomStrip-%s.png" % index)

        # use the average brightness to determine
        if get_average_brightness(topStrip) > get_average_brightness(bottomStrip):
            # use the bottom section, strip off top
            # print"using bottom"
            pilImage = pilImage.crop((0, stripHeight, width, height))
        else:
            # print"using top"
            # use the top section, strip off bottom
            pilImage = pilImage.crop((0, 0, width, height - stripHeight))

    elif aspectRatio > largeAspectRatioThreshold:
        # if its a large game piece, strip off both top and bottom strips
        pilImage = pilImage.crop((0, height * multiStripDelta, width, height - height * multiStripDelta))

    pilImage.save("output-images/5-strip-cropped-%s.png" % index)

    # print'original dimensions:', width, height

    if index == 0:
        rando = pilImage
    # proportional crop positions for getting the online code
    width, height = pilImage.size
    left = int(width * .10)
    top = int(height * .64)
    right = int(width * .45)
    bottom = int(height * .78)
    # print'online code crop positions:', left, top, right, bottom

    onlineCodeImage = pilImage.crop((left, top, right, bottom))
    croppedWidth, croppedHeight = onlineCodeImage.size
    # print'online code cropped dimensions:', croppedWidth, croppedHeight
    desiredRatio = croppedWidth / DESIRED_COMBINED_CODE_WIDTH
    newHeight = int(croppedHeight / desiredRatio)
    onlineCodeImage = onlineCodeImage.resize((DESIRED_COMBINED_CODE_WIDTH, newHeight), Image.ANTIALIAS)
    onlineCodeImage.save("output-images/6-online-code-cropped-%s.png" % index)
    onlineCodeImages.append(onlineCodeImage)

    # proportional crop positions for getting the manual game pieces
    top = int(height * 1)
    right = int(width * 1)
    bottom = int(height * 0)
    left = int(width * .57)
    # print'manual code crop positions:', left, top, right, bottom

    manualCodeImage = pilImage.crop((left, bottom, right, top))
    croppedWidth, croppedHeight = manualCodeImage.size

    # save individual game pieces
    manual_1 = manualCodeImage.crop((0, 0, croppedWidth * 0.25, croppedHeight))
    manual_1.save("output-images/7-b1-manual-code-cropped-%s.png" % index)
    manual_2 = manualCodeImage.crop((croppedWidth * 0.25, 0, croppedWidth * 0.5, croppedHeight))
    manual_2.save("output-images/7-b2-manual-code-cropped-%s.png" % index)
    manual_3 = manualCodeImage.crop((croppedWidth * 0.5, 0, croppedWidth * 0.75, croppedHeight))
    manual_3.save("output-images/7-b3-manual-code-cropped-%s.png" % index)
    manual_4 = manualCodeImage.crop((croppedWidth * 0.75, 0, croppedWidth, croppedHeight))
    manual_4.save("output-images/7-b4-manual-code-cropped-%s.png" % index)
    
    
    # print'manual code cropped dimensions:', croppedWidth, croppedHeight
    desiredRatio = croppedWidth / DESIRED_COMBINED_CODE_WIDTH
    newHeight = int(croppedHeight / desiredRatio)
    manualCodeImage = manualCodeImage.resize((DESIRED_COMBINED_CODE_WIDTH, newHeight), Image.ANTIALIAS)
    manualCodeImage.save("output-images/7-manual-code-cropped-%s.png" % index)
    manualCodeImages.append(manualCodeImage)

onlineWidths, onlineHeights = zip(*(i.size for i in onlineCodeImages))
manualWidths, manualHeights = zip(*(i.size for i in manualCodeImages))

# create a new image to place all the online codes on
totalWidth = max(onlineWidths)
maxHeight = sum(onlineHeights)
combinedOnlineCodesImage = Image.new('RGB', (totalWidth, maxHeight), (0, 0, 0, 255))
yOffset = 0
for index, img in enumerate(onlineCodeImages):
    combinedOnlineCodesImage.paste(img, (0, yOffset))
    yOffset += img.size[1]
# combinedOnlineCodesImage.show()
combinedOnlineCodesImage.save('output-images/8-online-codes-together.png')

# create a new image to place all the manual codes on
bufferHeight = 5
totalWidth = max(manualWidths)
maxHeight = sum(manualHeights) + bufferHeight * len(manualHeights)
combinedManualCodesImage = Image.new('RGB', (totalWidth, maxHeight), (178, 197, 217, 255))
yOffset = 0
for index, img in enumerate(manualCodeImages):
    combinedManualCodesImage.paste(img, (0, yOffset))
    yOffset += img.size[1] + bufferHeight

#####################################################
combinedManualCodesImage.show()
new = Image.new('RGB', (298, 233), (178, 197, 217, 255))
col1 = combinedManualCodesImage.crop((0, 0, 69, 233))
col2 = combinedManualCodesImage.crop((107, 0, 176, 233))
col3 = combinedManualCodesImage.crop((218, 0, 287, 233))
col4 = combinedManualCodesImage.crop((325, 0, 400, 233))
new.paste(col1, (0, 0))
new.paste(col2, (74, 0))
new.paste(col3, (148, 0))
new.paste(col4, (223, 0))