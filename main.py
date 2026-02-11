from PIL import Image
import cv2
from matplotlib import pyplot as plt

image_file = "data/download.jpg"
img = cv2.imread(image_file)

# cv2.imshow("Original image", img)
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

thresh, im_bw = cv2.threshold(gray_image, 150, 170, cv2.THRESH_BINARY)
cv2.imwrite("temp/im_bw.jpg", im_bw)

def noise_removal(image):
     import numpy as np
     kernel = np.ones((1, 1), np.uint8)
     image = cv2.dilate(image, kernel, iterations=1)
     kernel = np.ones((1, 1), np.uint8)
     image = cv2.erode(image, kernel, iterations=1)
     image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
     image = cv2.medianBlur(image, 3)
     return image

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
display("temp/dilated_image.jpg")


new = cv2.imread("data/download.jpg")
