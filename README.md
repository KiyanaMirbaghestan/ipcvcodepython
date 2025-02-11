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

# upload picture
image = Image.open(r"Desktop\JI230816Cosmos220-6d9254f-edited-scaled.jpg")

#Does the picture upload successfully?

if image is not None:
    # reverse pic
    im_flip = ImageOps.flip(image)
    
    # show the reversing pic
    im_flip.show()

    # save
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


#Crop picture/ first consider the dimensions

import cv2

image_path = "C:\\Users\\NoteBook\\Desktop\\JI230816Cosmos220-6d9254f-edited-scaled.jpg"
image = cv2.imread(image_path)

if image is None:
    print("تصویر بارگذاری نشد! مسیر را بررسی کنید.")
else:
    print("تصویر با موفقیت بارگذاری شد!")
    print(f"ابعاد تصویر: {image.shape}")  # نمایش اندازه تصویر


تصویر با موفقیت بارگذاری شد!
ابعاد تصویر: (1920, 2560, 3)




import cv2

image_path = "C:\\Users\\NoteBook\\Desktop\\JI230816Cosmos220-6d9254f-edited-scaled.jpg"
image = cv2.imread(image_path)
upper=150
lower=400
crop_top=image[upper:lower,:,:]
left=150
right=400
crop_horizontal=crop_top[:,left:right,:]
cv2.imshow("Cropped Image", crop_horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()


![cropped_image](https://github.com/user-attachments/assets/abe6768d-beda-43be-9bc7-46884af77c44)



##################we now want to write some special things. For example, a square or a text.

import cv2
from PIL import Image, ImageDraw, ImageFont
import os

print("🔍 شروع برنامه...")

# خواندن تصویر
image_path = "C:\\Users\\NoteBook\\Desktop\\JI230816Cosmos220-6d9254f-edited-scaled.jpg"
image = cv2.imread(image_path)

if image is None:
    print("❌ تصویر بارگذاری نشد! مسیر را بررسی کنید.")
    exit()

print("✅ تصویر با موفقیت بارگذاری شد.")


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)


image_draw = image_pil.copy()  # اینجا متغیر image_draw تعریف می‌شود
draw = ImageDraw.Draw(image_draw)


left, upper, right, lower = 150, 150, 400, 400
shape = [left, upper, right, lower]
draw.rectangle(xy=shape, outline="red", width=5)


try:
    font_path = r"C:\Windows\Fonts\arial.ttf"
    font = ImageFont.truetype(font_path, 50)
    draw.text((left, upper - 50), "box", font=font, fill="red")
    print("✅ متن اضافه شد.")
except Exception as e:
    print(f"⚠️ خطا در بارگذاری فونت: {e}")


image_draw.show()  # نمایش تصویر
print("📷 تصویر نمایش داده شد.")


save_path = "C:\\Users\\Public\\image_with_box.jpg"
try:
    image_draw.save(save_path)  # ذخیره تصویر
    print(f"✅ تصویر ذخیره شد: {save_path}")
except Exception as e:
    print(f"❌ خطا در ذخیره تصویر: {e}")


if os.path.exists(save_path):
    print(f"✅ فایل در مسیر ذخیره شده است: {save_path}")
else:
    print("❌ تصویر ذخیره نشد!")


![Shot 0015](https://github.com/user-attachments/assets/8bfcffbe-381f-472c-9e65-28cebf3e3715)





##############################This project will continue............
