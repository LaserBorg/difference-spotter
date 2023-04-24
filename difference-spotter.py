import cv2
import numpy as np
import os

# filter settings
max_width = 1024
min_area = 20
blur_diameter = 7
median_diameter = 3
mask_threshold = 9

# paths
img1_path = 'images/A.jpg'
img2_path = 'images/B.jpg'
output_path = 'outputs'


img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Scale both input images down to max width
if img1.shape[1] > max_width or img2.shape[1] > max_width:
    scale_percent = max_width / max(img1.shape[1], img2.shape[1])
    width = int(img1.shape[1] * scale_percent)
    height = int(img1.shape[0] * scale_percent)
    dim = (width, height)
    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)


# blend image 1 and 2 for preview
img_blend = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)

# convert a copy of the images to float
img1f = img1.astype(float) / 255
img2f = img2.astype(float) / 255

img1f = cv2.GaussianBlur(img1f, (blur_diameter, blur_diameter), 0)
img2f = cv2.GaussianBlur(img2f, (blur_diameter, blur_diameter), 0)

# Split both input images into separate images per channel
bgr_planes1 = cv2.split(img1f)
bgr_planes2 = cv2.split(img2f)

# Subtract R_1 - R_2, G_1 - G_2, B_1 - B_2
difference_planes = []
for plane1, plane2 in zip(bgr_planes1, bgr_planes2):
    difference_planes.append(cv2.subtract(plane1, plane2))

# Divide each image by 3 before adding
mask = difference_planes[0] / 3 + difference_planes[1] / 3 + difference_planes[2] / 3

# Get the absolute value of the difference image and clip values
mask = np.abs(mask)
mask = np.clip(mask, 0, 1)

# median filtering the mask
mask = cv2.medianBlur(mask.astype('float32'), median_diameter)

# convert back to uint8
mask = (mask * 255).astype('uint8')

# binarize the mask
_, mask_binarized = cv2.threshold(mask, mask_threshold, 255, cv2.THRESH_BINARY)

# Find contours in mask and draw rectangles around each contour if its size is bigger than 10 pixels
contours, _ = cv2.findContours(mask_binarized.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) > min_area:
        x,y,w,h = cv2.boundingRect(contour)
        crop_img1 = img1[y:y+h,x:x+w]
        crop_img2 = img2[y:y+h,x:x+w]
        vis_img = np.concatenate((crop_img1,crop_img2), axis=1)

        # save concatenated ROIs as jpg
        filename = os.path.join(output_path, str(x) + '_' + str(y) + '.jpg')
        cv2.imwrite(filename, vis_img)

        # draw rectangles in preview-image
        cv2.rectangle(img_blend, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.imshow('Bounding Boxes', img_blend)
cv2.imwrite(os.path.join(output_path, "bounding_boxes.jpg"), img_blend)

cv2.waitKey(0)
cv2.destroyAllWindows()
