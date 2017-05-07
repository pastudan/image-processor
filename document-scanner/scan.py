# USAGE
# python scan.py --image images/page.jpg 

# import the necessary packages
from __future__ import division
from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive
from math import sqrt
from PIL import Image, ImageOps, ImageEnhance
from google.cloud import vision
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
cv2.imwrite('1-original.png', orig)
cv2.imwrite('1-edges.png', edged)

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
cv2.imwrite('2-contours.png', image)

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

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # warped = threshold_adaptive(warped, 251, offset=10)
    # warped = warped.astype("uint8") * 255

    # STEP 3: Apply perspective transform
    cv2.imwrite("3-warped-%s.png" % index, warped)

    # STEP 4: Crop only desired areas
    pilImage = Image.fromarray(warped)
    width, height = pilImage.size

    aspectRatio = height / width
    if smallAspectRatioThreshold < aspectRatio < largeAspectRatioThreshold:
        stripHeight = singleStripDelta * height * 1.3

        topStrip = pilImage.crop((0, 0.1 * stripHeight, width, stripHeight))
        bottomStrip = pilImage.crop((0, height - stripHeight, width, height - 0.1 * stripHeight))

        # topStrip.save("4-topStrip-%s.png" % index)
        # bottomStrip.save("4-bottomStrip-%s.png" % index)

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

    pilImage.save("5-strip-cropped-%s.png" % index)

    # print'original dimensions:', width, height

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
    onlineCodeImage.save("6-online-code-cropped-%s.png" % index)
    onlineCodeImages.append(onlineCodeImage)

    # proportional crop positions for getting the manual game pieces
    left = int(width * .57)
    top = int(height * .64)  # .64
    right = width
    bottom = int(height * .78)  # .77
    # print'manual code crop positions:', left, top, right, bottom

    manualCodeImage = pilImage.crop((left, top, right, bottom))
    croppedWidth, croppedHeight = manualCodeImage.size
    # print'manual code cropped dimensions:', croppedWidth, croppedHeight
    desiredRatio = croppedWidth / DESIRED_COMBINED_CODE_WIDTH
    newHeight = int(croppedHeight / desiredRatio)
    manualCodeImage = manualCodeImage.resize((DESIRED_COMBINED_CODE_WIDTH, newHeight), Image.ANTIALIAS)
    # manualCodeImage = manualCodeImage.convert('L')  #convert to grayscale
    # grayscaleImageArray = threshold_adaptive(numpy.asarray(manualCodeImage), 251, offset=10)
    # manualCodeImage = Image.fromarray(grayscaleImageArray.astype("uint8") * 255)
    # manualCodeImage = ImageOps.invert(manualCodeImage)
    # manualCodeImage = ImageEnhance.Contrast(manualCodeImage).enhance(1.5)
    manualCodeImage.save("7-manual-code-cropped-%s.png" % index)
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
combinedOnlineCodesImage.save('8-online-codes-together.png')

# create a new image to place all the manual codes on
bufferHeight = 0
totalWidth = max(manualWidths)
maxHeight = sum(manualHeights) + bufferHeight * len(manualHeights)
combinedManualCodesImage = Image.new('RGB', (totalWidth, maxHeight), (179, 214, 243, 255))
yOffset = 0
for index, img in enumerate(manualCodeImages):
    combinedManualCodesImage.paste(img, (0, yOffset))
    yOffset += img.size[1] + bufferHeight
combinedManualCodesImage.show()
combinedManualCodesImage.save('8-manual-codes-together.png')

vision_client = vision.Client()

# # load as png into memory and send to google
# imgByteArr = io.BytesIO()
# combinedOnlineCodesImage.save(imgByteArr, format='PNG')
# image = vision_client.image(content=imgByteArr.getvalue())
# onlineDocument = image.detect_text()

# load as png into memory and send to google
imgByteArr = io.BytesIO()
combinedManualCodesImage.save(imgByteArr, format='PNG')
image = vision_client.image(content=imgByteArr.getvalue())
manualDocument = image.detect_text()

# if not len(onlineDocument) or not len(manualDocument):
#     exit(1)
#
# onlineText = onlineDocument[0].description
manualText = manualDocument[0].description
# if not onlineText or not manualText:
#     exit(2)

# onlineLines = onlineText.splitlines()
manualLines = manualText.splitlines()

correctOnlineCodes = [
    'M7C9N9C5Z78X462K',
    'MHT78WCLV7NP1G7C',
    'M2G9YPCN57XC7WTH',
    'MKGCWLCMC7ZXV73T',
    'MA9PCXCTR7552GNK',
    'M8ZLWTC5L7V6A235',
    'MH7T85CXW7GH9RMC',
]

correctManualCodes = [
    '8H69A',
    '8T30B',
    '8V25A',
    '8Y15G',

    '8Z04D',
    '8H71C',
    '8B96A',
    '8Q42B',

    '8G76D',
    '9H33B',
    '8R38B',
    '8C95E',

    '8B97B',
    '9A02A',
    '9D18C',
    '8S33A',

    '9F25B',
    '8F79C',
    '8D88C',
    '8Z01A',

    '8Y15G',
    '9B09C',
    '8W23C',
    '8H69A',

    '8G73A',
    '8Y09A',
    '8P48D',
    '8H69A',

    '8V28D',
    '8Y10B',
    '9J36A',
    '8D90E',
]

onlineCodes = []
manualCodes = []

regex = re.compile('[^1-9ACGHKLMNPRTVWXYZ]')  # regex for valid game piece characters
# for index, line in enumerate(onlineLines):
#     line = regex.sub('', line)  # remove invalid characters
#     onlineCodes.append(line)

for line in manualLines:
    manualCodes += line.split()

from pprint import pprint

# pprint(correctOnlineCodes)
# pprint(onlineCodes)
# 
# pprint(manualLines)
# pprint(manualCodes)

correctOnlineCount = sum(True for code in onlineCodes if code in correctOnlineCodes)
correctManualCount = sum(True for code in manualCodes if code in correctManualCodes)

# print"%s / %s correct online codes (%s%%)" % (correctOnlineCount, len(correctOnlineCodes), correctOnlineCount / len(correctOnlineCodes) * 100)
print"%s / %s correct manual codes (%s%%)" % (
    correctManualCount, len(correctManualCodes), correctManualCount / len(correctManualCodes) * 100)


# import ipdb; ipdb.set_trace()
