#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from google.colab.patches import cv2_imshow 


# In[ ]:


from google.colab import files
from IPython.display import Image


# In[ ]:


uploaded = files.upload()
     


# In[ ]:


image = cv2.imread("images.jpg")


# In[ ]:


cv2_imshow(image)
     


# In[ ]:





# Grayscale and Convert

# In[ ]:


grey_filter = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2_imshow(grey_filter)


# In[ ]:


invert =cv2.bitwise_not(grey_filter)
cv2_imshow(invert)


# In[ ]:





# Blur Image

# In[ ]:


blur_image=cv2.GaussianBlur(invert,(21,21),0)
cv2_imshow(blur_image)


# In[ ]:





# Pencil Sketch 

# In[ ]:


invertedblur_image=cv2.bitwise_not(blur_image)
cv2_imshow(invertedblur_image)
Pencil_sketch_filter_image=cv2.divide(grey_filter,invertedblur_image,scale=256.0)
cv2_imshow(Pencil_sketch_filter_image)


# In[ ]:


image = cv2.imread("download.jpg")
cv2_imshow(image)
grey_filter = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2_imshow(grey_filter)
invert =cv2.bitwise_not(grey_filter)
cv2_imshow(invert)
blur_image=cv2.GaussianBlur(invert,(21,21),0)
cv2_imshow(blur_image)
invertedblur_image=cv2.bitwise_not(blur_image)
cv2_imshow(invertedblur_image)
Pencil_sketch_filter_image=cv2.divide(grey_filter,invertedblur_image,scale=256.0)
cv2_imshow(Pencil_sketch_filter_image)


# In[ ]:




