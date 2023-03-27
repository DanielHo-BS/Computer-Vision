import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.float32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        # Pre-compute
        h, w, ch = img.shape
        output = np.zeros_like(img).astype(np.float32)
        scaleFactor_s = 1 / (2 * (self.sigma_s ** 2))
        scaleFactor_r = 1 / (2 * (self.sigma_r ** 2))
        # A lookup table for range kernel
        LUT = np.exp(-np.arange(256) / 255 * np.arange(256) / 255 * scaleFactor_r)
        # A spatial Gaussian function
        x ,y = np.meshgrid(np.arange(self.wndw_size) - self.pad_w , np.arange(self.wndw_size) - self.pad_w)
        kernel_s = np.exp(-(x ** 2 + y ** 2) * scaleFactor_s)
        
        # Main
        if padded_img.ndim == 3 and padded_guidance.ndim == 3:   # RGB-image and RGB-guidance
            for y in range(self.pad_w, self.pad_w + h):
                for x in range(self.pad_w, self.pad_w + w):
                    # Range kernel * Spaial kernel
                    wgt = LUT[abs(padded_guidance[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 0] - padded_guidance[y, x, 0])] * \
                          LUT[abs(padded_guidance[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 1] - padded_guidance[y, x, 1])] * \
                          LUT[abs(padded_guidance[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 2] - padded_guidance[y, x, 2])] * \
                          kernel_s
                    wacc = np.sum(wgt)
                    # Convolution
                    output[y - self.pad_w, x - self.pad_w, 0] = np.sum(wgt * padded_img[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 0]) / wacc
                    output[y - self.pad_w, x - self.pad_w, 1] = np.sum(wgt * padded_img[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 1]) / wacc
                    output[y - self.pad_w, x - self.pad_w, 2] = np.sum(wgt * padded_img[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 2]) / wacc
        elif padded_img.ndim == 3 and padded_guidance.ndim == 2:   # RGB-image and gray-guidance
            for y in range(self.pad_w, self.pad_w + h):
                for x in range(self.pad_w, self.pad_w + w):
                    # Range kernel * Spaial kernel
                    wgt = LUT[abs(padded_guidance[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1] - padded_guidance[y, x])] * kernel_s
                    wacc = np.sum(wgt)
                    # Convolution
                    output[y - self.pad_w, x - self.pad_w, 0] = np.sum(wgt * padded_img[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 0]) / wacc
                    output[y - self.pad_w, x - self.pad_w, 1] = np.sum(wgt * padded_img[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 1]) / wacc
                    output[y - self.pad_w, x - self.pad_w, 2] = np.sum(wgt * padded_img[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 2]) / wacc
        elif padded_img.ndim == 2 and padded_guidance.ndim == 3:   # gray-image and RGB-guidance
            for y in range(self.pad_w, self.pad_w + h):
                for x in range(self.pad_w, self.pad_w + w):
                    # Range kernel * Spaial kernel
                    wgt = LUT[abs(padded_guidance[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 0] - padded_guidance[y, x, 0])] * \
                          LUT[abs(padded_guidance[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 1] - padded_guidance[y, x, 1])] * \
                          LUT[abs(padded_guidance[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1, 2] - padded_guidance[y, x, 2])] * \
                          kernel_s
                    wacc = np.sum(wgt)
                    # Convolution
                    output[y - self.pad_w, x - self.pad_w] = np.sum(wgt * padded_img[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1]) / wacc
        elif padded_img.ndim == 2 and padded_guidance.ndim == 2:  # gray-image and gray-guidance
            for y in range(self.pad_w, self.pad_w + h):
                for x in range(self.pad_w, self.pad_w + w):
                    # Range kernel * Spaial kernel
                    wgt = LUT[abs(padded_guidance[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1] - padded_guidance[y, x])] * kernel_s
                    wacc = np.sum(wgt)
                    # Convolution
                    output[y - self.pad_w, x - self.pad_w] = np.sum(wgt * padded_img[y - self.pad_w:y + self.pad_w + 1, x - self.pad_w:x + self.pad_w + 1]) / wacc 
        else:
            print('Error ndim of image or guidance!!!')
        return np.clip(output, 0, 255).astype(np.uint8)