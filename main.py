from PIL import Image
import cv2
from matplotlib import pyplot as plt
import pytesseract

image_file = "data/download.jpg"
img = cv2.imread(image_file)


cv2.waitKey(0)
def display(im_path): #from stackoverflow
     dpi = 80
     im_data = plt.imread(im_path)
     if len(im_data.shape) == 3:
          height, width, depth = im_data.shape
     else:
          height, width = im_data.shape

     figsize = width / float(dpi), height / float(dpi)

     fig = plt.figure(figsize=figsize)
     ax = fig.add_axes([0,0,1,1])
     ax.axis('off')
     ax.imshow(im_data, cmap='gray')
     plt.show()

inverted_image = cv2.bitwise_not(img)
cv2.imwrite("temp/inverted.jpg", inverted_image)


def grayscale(image):

     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = grayscale(img)
cv2.imwrite("temp/gray.jpg", gray_image)

# thresh, im_bw = cv2.threshold(gray_image, 190, 205, cv2.THRESH_BINARY)
thresh, im_bw = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("temp/im_bw.jpg", im_bw)
display("temp/im_bw.jpg")


def noise_removal(image):
     # import numpy as np
     # kernel = np.ones((1, 1), np.uint8)
     # image = cv2.dilate(image, kernel, iterations=1)
     # kernel = np.ones((1, 1), np.uint8)
     # image = cv2.erode(image, kernel, iterations=1)
     # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
     # image = cv2.medianBlur(image, 3)
     # return image
     return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

no_noise = noise_removal(im_bw)
cv2.imwrite("temp/no_noise.jpg", no_noise)

def thin_font(image):
     import numpy as np
     image = cv2.bitwise_not(image)
     kernel = np.ones((2, 2), np.uint8)
     image = cv2.erode(image, kernel, iterations=1)
     image = cv2.bitwise_not(image)
     return image

eroded_image = thin_font(no_noise)
cv2.imwrite("temp/eroded_image.jpg", eroded_image)


def thick_font(image):
     import numpy as np
     image = cv2.bitwise_not(image)
     kernel = np.ones((2, 2), np.uint8)
     image = cv2.dilate(image, kernel, iterations=1)
     image = cv2.bitwise_not(image)
     return image

dilated_image = thick_font(eroded_image)
cv2.imwrite("temp/dilated_image.jpg", dilated_image)
# display("temp/dilated_image.jpg")


new = cv2.imread("data/download.jpg")

#https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
import numpy as np

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle
# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

fixed = deskew(new)
cv2.imwrite("temp/rotated_fixed.jpg", fixed)

# display("temp/rotated_fixed.jpg")

def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)
no_borders = remove_borders(no_noise)
cv2.imwrite("temp/no_borders.jpg", no_borders)
# display('temp/no_borders.jpg')

color = [255, 255, 255]
top, bottom, left, right = [150]*4
image_with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
cv2.imwrite("temp/image_with_border.jpg", image_with_border)
display("temp/image_with_border.jpg")

final_img = Image.open("temp/image_with_border.jpg")
ocr_result = pytesseract.image_to_string(final_img)
print(ocr_result)