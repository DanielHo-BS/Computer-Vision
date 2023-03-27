import numpy as np
import cv2
import os

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image, save_images = False):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)

        # Init
        gaussian_images = []
        gaussian_images_resize = []
        image_shape = image.shape

        # Octave 1
        gaussian_images = [cv2.GaussianBlur(image.copy(),(0, 0), self.sigma**k).copy() for k in range(1,self.num_guassian_images_per_octave)]
        gaussian_images.insert(0,image.copy())

        # Octave 2
        image_resize = cv2.resize(gaussian_images[4], (image_shape[1]//2, image_shape[0]//2), interpolation=cv2.INTER_NEAREST).copy()
        gaussian_images_resize = [cv2.GaussianBlur(image_resize.copy(),(0, 0), self.sigma**k).copy() for k in range(1,self.num_guassian_images_per_octave)]
        gaussian_images_resize.insert(0,image_resize.copy())


        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        
        # Init
        dog_images = []
        dog_images_resise = []

        # Subtract
        dog_images = [((cv2.subtract(gaussian_images[j+1], gaussian_images[j])).copy()) for j in range(self.num_DoG_images_per_octave)]
        dog_images_resise = [((cv2.subtract(gaussian_images_resize[j+1], gaussian_images_resize[j])).copy()) for j in range(self.num_DoG_images_per_octave)]
        
        # Save DoG images
        def saveResult(img,keyword):
            if save_images:
                if not os.path.exists("out"):
                    os.mkdir("out")
                for i in range(len(img)):
                    out = img[i].copy()
                    out = ((out - out.min()) / (out.max()-out.min()) *255).round()
                    cv2.imwrite("out/" + keyword + str(i+1) +".png", out)

        saveResult(dog_images, "DoG1-")
        saveResult(dog_images_resise, "DoG2-")
        

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        
        # Turn List to image
        dog_images = cv2.merge([dog_images[1],dog_images[2],dog_images[3]]).copy()
        dog_images_resise = cv2.merge([dog_images_resise[1],dog_images_resise[2],dog_images_resise[3]]).copy()
        
        # Find local extremum 
        def findExtremum(image):
            shape = image.shape
            result = np.zeros([shape[0],shape[1],1])
            for i in range(1,shape[0]):
                for j in range(1,shape[1]):
                    if image[i,j,1] <= image[i-1:i+2,j-1:j+2,:].min() :
                        result[i,j,0] =  image[i,j,1]
                    elif image[i,j,1] >= image[i-1:i+2,j-1:j+2,:].max() :
                            result[i,j,0] =  image[i,j,1]       

            return result        
    
        extremum_images = findExtremum(dog_images)
        extremum_images_resize = findExtremum(dog_images_resise)
        
        # Thresholding
        def threshold(image):
            img = abs(image[:,:,0].copy())
            array =  []
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j] >= self.threshold:
                        array.append((i,j))
                        
            return np.array(array)
        
        threshold_points = threshold(extremum_images)
        threshold_points_resize = threshold(extremum_images_resize) * 2
        if threshold_points_resize != []:
            local_extremum = np.concatenate((threshold_points,threshold_points_resize),axis=0)
        else: # Fix error concatenate with []
            local_extremum = np.concatenate((threshold_points,threshold_points),axis=0)
            
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(local_extremum,axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
