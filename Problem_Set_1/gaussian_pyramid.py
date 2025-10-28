import numpy as np
from PIL import Image

def cross_correlation_2d(img, knl):
    img_h, img_w = img.shape
    knl_h, knl_w = knl.shape
    pad_h = knl_h // 2
    pad_w = knl_w // 2
    img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 
        mode='constant', constant_values=0)
    ans = np.zeros_like(img) 

    for i in range(img_h):
        for j in range(img_w):
            region = img[i:i+knl_h, j:j+knl_w]
            ans[i, j] = np.sum(region * knl)
    return ans

def convolve_2d(img, knl):
    knl = np.flip(knl, 0)
    knl = np.flip(knl, 1)
    return cross_correlation_2d(img, knl)

def gaussian_blur_kernel_2d(size=3, sigma=1):
    ans = np.zeros((size, size), dtype=np.float32)
    r = size // 2
    
    for i in range(size):
        for j in range(size):
            x = i - r
            y = j - r
            ans[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    ans /= np.sum(ans)
    return ans

def low_pass(img, size=3, sigma=1):
    knl = gaussian_blur_kernel_2d(size, sigma)
    img_array = np.array(img)
    shape = img_array.shape[:2]

    red = img_array[:, :, 0].reshape(shape)
    green = img_array[:, :, 1].reshape(shape)
    blue = img_array[:, :, 2].reshape(shape)
    red_blurred = convolve_2d(red, knl)
    green_blurred = convolve_2d(green, knl)
    blue_blurred = convolve_2d(blue, knl)

    ans = np.stack((red_blurred, green_blurred, blue_blurred), axis=-1)
    return ans

def image_subsampling(img):
    return img[::2,::2,:]

def gaussian_pyramid(img, size=3, sigma=1):
    for i in range(1, 4):
        blurred_img = low_pass(img, size, sigma)    
        subsampled_img = image_subsampling(blurred_img)
        img = Image.fromarray(subsampled_img.astype(np.uint8))
        img.save(f"./{name}{i}.png", "PNG")

if __name__=="__main__":
    name = input("请输入将要处理的图像的名字（不带后缀）: ")
    image = Image.open(f"./{name}.png")
    gaussian_pyramid(image, 3, 1)