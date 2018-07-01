 #!/usr/bin/python
import matplotlib
#matplotlib.use("qt4agg")
#import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.ndimage as ndimage

class plane_clustering:
    def __init__(self, **kwargs):
        self.mat_path = kwargs.get('mat_path', '.')
        self.smoothed = kwargs.get('smoothed', False)
        self.sigm = kwargs.get('sigma', 1)
        self.n = 0
        if not self.__load_mat():
            print('Error loading \'mat\' dataset')
            exit()
        self.var_names = ('accelData', 'depths', 'rawDepths', 'images', 'instances',
                          'labels', 'names', 'namesToIds', 'sceneTypes', 'scenes')
        self.current_index = -1
        self.next()

    def next(self):
        if self.current_index+1 >= self.n:
            print('Finished')
            exit()
        self.current_index = self.current_index+1
        self.image = self.dataset_dict['images'][:,:,:,self.current_index]
        shap = self.image.shape
        self.H = shap[0]
        self.W = shap[1]
        self.depth = self.dataset_dict['depths'][:,:,self.current_index]
        if self.smoothed:
            self.depth = ndimage.gaussian_filter(self.depth,sigma=(self.sigm,self.sigm),order=0)
        self.label = self.dataset_dict['labels'][:,:,self.current_index]
        self.point_cloud = np.zeros((3, self.H, self.W))
        self.__depth2xyz()
        print('data loaded successfully')

    def __load_mat(self):
        try:
            self.dataset_dict = sio.loadmat(self.mat_path)
            self.n = self.dataset_dict['depths'][1,1,:].size
            return True
        except:
            pass
        return False

    def __depth2xyz(self):
        depth = self.depth
        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02
        [xx, yy] = np.meshgrid(range(1,self.W+1),range(1,self.H+1))
        self.point_cloud[0,:,:] = (xx - cx_rgb) * depth / fx_rgb
        y = (yy - cy_rgb) * depth / fy_rgb
        y = -y+np.ndarray.max(y)
        self.point_cloud[1,:,:] = depth
        self.point_cloud[2,:,:] = y



if __name__ == "__main__":
    #pc = plane_clustering(mat_path='../../3DFinalProj/mat/reduced_dataset.mat')
    pc = plane_clustering(mat_path='../reduced_reduced_dataset.mat',smoothed=True, sigma=0.8)
    
    
    #print(pc.point_cloud)
    #plt.imshow(pc.image, interpolation='nearest')
    #plt.show()
    #plt.imshow(ndimage.gaussian_filter(pc.image,sigma=(1,1,0),order=0))
    #plt.show()