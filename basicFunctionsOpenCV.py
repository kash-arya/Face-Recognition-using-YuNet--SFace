import cv2

IMAGE_PATH = 'assets/test1.jpeg'
GRAY_IMAGE_PATH = 'assets/test1_gray.jpeg'

# REMEMBER: cv2.imread() loads an image from a file path.
# If the path is wrong or the file cannot be read, it returns None.
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# REMEMBER: OpenCV loads color images in BGR order, not RGB.
# cv2.cvtColor() is used to convert between color formats.
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# REMEMBER: cv2.imwrite() saves an image and returns True or False.
saved = cv2.imwrite(GRAY_IMAGE_PATH, gray_image)
print(f"Grayscale image saved: {saved}")

# REMEMBER: cv2.imshow() displays an image in a separate window.
cv2.imshow('Original', image)
cv2.imshow('Grayscale', gray_image)

# REMEMBER: waitKey(0) keeps the windows open until any key is pressed.
# destroyAllWindows() closes all OpenCV display windows.
cv2.waitKey(0)
cv2.destroyAllWindows()

# REMEMBER: image.shape gives image dimensions.
# Color image shape is usually (height, width, channels).
# Grayscale image shape is usually (height, width).
print(f"Original dimensions: {image.shape}")
print(f"Grayscale dimensions: {gray_image.shape}")
