# ipcvcodepython
I am learning image processing and computer vision. This is a simple project from me.
#Install and import required packages:
pip install pillow
import PIL
print(PIL.__version__)
from PIL import Image

#Having a black-and-white image
image=Image.open("Desktop/JI230816Cosmos220-6d9254f-edited-scaled.jpg")
image.show()
![Flipped_Image](https://github.com/user-attachments/assets/c69bcc8c-958b-4feb-a2b2-95a0b66bea71)

#we have pixels
import matplotlib.pyplot as plt
plt.imshow(image)
![output_4_1](https://github.com/user-attachments/assets/b46c1802-d5c6-4894-89f5-594a71ad5f01)

#Having black-and-white image 
from PIL import ImageOps
image_gray=ImageOps.grayscale(image)
image_gray.quantize(2)

![output_6_0](https://github.com/user-attachments/assets/ebe15283-9d55-4f7e-9a28-ca60c4e4a5f6)


#converting to array
import numpy as np
from PIL import Image

# Open the image
im = Image.open("Desktop/JI230816Cosmos220-6d9254f-edited-scaled.jpg")

# Convert the image to a NumPy array
arrays = np.array(im)

# Print the array
print(arrays)

[[193 193 193 ... 215 215 215]
 [193 193 193 ... 215 215 215]
 [193 193 193 ... 215 215 215]
 ...
 [ 95  94  94 ... 103 103 103]
 [ 95  95  95 ... 103 103 103]
 [ 96  96  96 ... 103 103 103]]



#####work with opencv
import cv2
import numpy
image=cv2.imread("Desktop/JI230816Cosmos220-6d9254f-edited-scaled.jpg")
type(image)
numpy.ndarray
import matplotlib.pyplot as plt 
plt.imshow(image)

![image](https://github.com/user-attachments/assets/a7e03b61-32f3-4206-8775-c22a63b1bd64)



new_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(new_image)
#or
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray, cmap='gray')  # Add cmap='gray' to properly display grayscale images
cv2.imwrite("Desktop/JI230816Cosmos220-6d9254f-edited-scaled.jpg", image_gray)

![image](https://github.com/user-attachments/assets/2161138b-2d18-4e85-9987-2790706936a6)



import cv2  # Correct import statement

# Read the image
im_gray = cv2.imread("Desktop/JI230816Cosmos220-6d9254f-edited-scaled.jpg")

blue, green, red = im_gray[:, :, 0], im_gray[:, :, 1], im_gray[:, :, 2]

print(blue, green, red)


[[193 193 193 ... 215 215 215]
 [193 193 193 ... 215 215 215]
 [193 193 193 ... 215 215 215]
 ...
 [ 95  94  94 ... 103 103 103]
 [ 95  95  95 ... 103 103 103]
 [ 96  96  96 ... 103 103 103]] [[193 193 193 ... 215 215 215]
 [193 193 193 ... 215 215 215]
 [193 193 193 ... 215 215 215]
 ...
 [ 95  94  94 ... 103 103 103]
 [ 95  95  95 ... 103 103 103]
 [ 96  96  96 ... 103 103 103]] [[193 193 193 ... 215 215 215]
 [193 193 193 ... 215 215 215]
 [193 193 193 ... 215 215 215]
 ...
 [ 95  94  94 ... 103 103 103]
 [ 95  95  95 ... 103 103 103]
 [ 96  96  96 ... 103 103 103]]




#######Rotate pictures


from PIL import ImageOps
from PIL import Image, ImageOps

# تصویر را بارگذاری کنید
image = Image.open(r"Desktop\JI230816Cosmos220-6d9254f-edited-scaled.jpg")

# بررسی کنید که تصویر بارگذاری شده است
if image is not None:
    # برعکس کردن تصویر
    im_flip = ImageOps.flip(image)
    
    # نمایش تصویر برعکس شده
    im_flip.show()

    # ذخیره تصویر برعکس شده
    im_flip.save("Desktop/Flipped_Image.png")
else:
    print("تصویر پیدا نشد. لطفاً مسیر فایل را بررسی کنید.")
from PIL import ImageOps
im_flip=ImageOps.flip(image)
im_flip.show()
from PIL import ImageOps
im_mirror=ImageOps.mirror(image)
im_mirror.show()
from PIL import ImageOps
image.transpose(Image.FLIP_TOP_BOTTOM)

![image](https://github.com/user-attachments/assets/afc9cfdd-9f9d-4739-b811-8f1fff426525)




##############################This project will continue............
