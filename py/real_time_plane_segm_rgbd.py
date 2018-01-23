 #!/usr/bin/python
import matplotlib
#matplotlib.use("qt4agg")
#import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.ndimage as sndim
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.vq import kmeans2, vq, whiten
from skimage import color


class plane_clustering:
    def __init__(self, **kwargs):
        self.current_index = -1
        self.normals = []
        self.clusters = []
        self.mat_path = kwargs.get('mat_path', '.')
        self.smoothed = kwargs.get('smoothed', False)
        self.sigm = kwargs.get('sigma', 1)
        self.n = 0
        if not self.__load_mat(): # self.dataset_dict, self.n
            print('Error loading \'mat\' dataset')
            exit()
        self.var_names = ('accelData', 'depths', 'rawDepths', 'images', 'instances',
                          'labels', 'names', 'namesToIds', 'sceneTypes', 'scenes')
        self.normals_length = 1
        # computing average normals
        self.avg_win_long = 1  # half of the greatest dimension of the avg window
        self.avg_win_short = 1 # half of the smaller dimension of the avg window
        self.next()

    def next(self):
        if self.current_index+1 >= self.n:
            print('Finished')
            return False
        self.current_index = self.current_index+1
        self.depth = self.dataset_dict['depths'][:, :, self.current_index]
        self.image = self.dataset_dict['images'][:, :, :, self.current_index]
        self.label = self.dataset_dict['labels'][:, :, self.current_index]
        self.H = self.image.shape[0]
        self.W = self.image.shape[1]
        if self.smoothed:
            self.depth = sndim.gaussian_filter(self.depth, sigma=(self.sigm, self.sigm), order=0)
        # from depth to [x,y,z] coordinates
        self.point_cloud = self.__depth2xyz()
        print('Image '+str(self.current_index+1)+' of '+str(self.n)+' loaded successfully')
        self.normals.append(self.compute_normals())
        self.clusters.append(self.run_kmeans())
        curr_cluster = self.clusters[self.current_index]
        #print(str(curr_cluster['normcolor'/all/xyz/color/normals]['centroid'/label]))
        
        return True
    
    def show(self):
        line_width = 0.4
        point_size = 0.5
        norm_length = 0.008
        lato = 30
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #ax = fig.add_subplot(111, projection='3d')
        current_pc = self.point_cloud
        current_norm = self.normals[self.current_index]
        current_pc = current_pc[:, 0:lato, 0:lato]
        current_norm = current_norm[:, 0:lato, 0:lato]
        for i in range(lato):
            for j in range(lato):
                leng = np.linalg.norm(current_norm[:, i, j])
                if (leng>0):
                    current_norm[:, i, j] = current_norm[:, i, j] / leng * norm_length
        #print(current_pc)
        #print(current_norm)
        ax.scatter(current_pc[0, :, :], current_pc[1, :, :], 
        current_pc[2, :, :], c='b',s=point_size)
        q = ax.quiver(current_pc[0, :, :], current_pc[1, :, :],current_pc[2, :, :],
        current_norm[0, :, :],current_norm[1, :, :],current_norm[2, :, :], lw=line_width)
        plt.draw()
        plt.show()
        return plt

    def compute_normals(self):
        H = self.H
        W = self.W
        tang_x = np.zeros((3,H,W))
        tang_y = np.zeros((3,H,W))
        for i in range(1,H-1):
            for j in range(1,W-1):
                left = self.point_cloud[:,i-1,j]
                right = self.point_cloud[:,i+1,j]
                up = self.point_cloud[:,i,j-1]
                down = self.point_cloud[:,i,j+1]
                tang_x[:,i,j] = right - left
                tang_y[:,i,j] = down - up
        integral_xx = np.zeros((H,W))
        integral_xy = np.zeros((H,W))
        integral_xz = np.zeros((H,W))
        integral_yx = np.zeros((H,W))
        integral_yy = np.zeros((H,W))
        integral_yz = np.zeros((H,W))
        for i in range(1,H):
            for j in range(1,W):
                integral_xx[i,j] = integral_xx[i-1,j]+integral_xx[i,j-1]-integral_xx[i-1,j-1]+tang_x[0,i,j]
                integral_xy[i,j] = integral_xy[i-1,j]+integral_xy[i,j-1]-integral_xy[i-1,j-1]+tang_x[1,i,j]
                integral_xz[i,j] = integral_xz[i-1,j]+integral_xz[i,j-1]-integral_xz[i-1,j-1]+tang_x[2,i,j]
                integral_yx[i,j] = integral_yx[i-1,j]+integral_yx[i,j-1]-integral_yx[i-1,j-1]+tang_y[0,i,j]
                integral_yy[i,j] = integral_yy[i-1,j]+integral_yy[i,j-1]-integral_yy[i-1,j-1]+tang_y[1,i,j]
                integral_yz[i,j] = integral_yz[i-1,j]+integral_yz[i,j-1]-integral_yz[i-1,j-1]+tang_y[2,i,j]
        print('integral image of tangents computed')
        # vectorial product between the 2 tangents to get the normal
        normals = np.zeros((3,H,W))
        valid = np.empty((H,W), bool)
        w1 = self.avg_win_long # half of the greatest dimension of the averaging window
        w2 = self.avg_win_short # half of the smaller dimension of the averaging window
        for i in range(w1,H-w1):
            for j in range(w1,W-w1):
                tan_xx_l = integral_xx[i+w1,j]-integral_xx[i+w1,j-w2]-integral_xx[i-w1,j]+integral_xx[i-w1,j-w2]
                tan_xx_r = integral_xx[i+w1,j+w2]-integral_xx[i+w1,j]-integral_xx[i-w1,j+w2]+integral_xx[i-w1,j]
                tan_xy_l = integral_xy[i+w1,j]-integral_xy[i+w1,j-w2]-integral_xy[i-w1,j]+integral_xy[i-w1,j-w2]
                tan_xy_r = integral_xy[i+w1,j+w2]-integral_xy[i+w1,j]-integral_xy[i-w1,j+w2]+integral_xy[i-w1,j]
                tan_xz_l = integral_xz[i+w1,j]-integral_xz[i+w1,j-w2]-integral_xz[i-w1,j]+integral_xz[i-w1,j-w2]
                tan_xz_r = integral_xz[i+w1,j+w2]-integral_xz[i+w1,j]-integral_xz[i-w1,j+w2]+integral_xz[i-w1,j]
                tan_yx_u = integral_yx[i,j+w1]-integral_yx[i,j-w1]-integral_yx[i-w2,j+w1]+integral_yx[i-w2,j-w1]
                tan_yx_d = integral_yx[i+w2,j+w1]-integral_yx[i,j+w1]-integral_yx[i+w2,j-w1]+integral_yx[i,j-w1]
                tan_yy_u = integral_yy[i,j+w1]-integral_yy[i,j-w1]-integral_yy[i-w2,j+w1]+integral_yy[i-w2,j-w1]
                tan_yy_d = integral_yy[i+w2,j+w1]-integral_yy[i,j+w1]-integral_yy[i+w2,j-w1]+integral_yy[i,j-w1]
                tan_yz_u = integral_yz[i,j+w1]-integral_yz[i,j-w1]-integral_yz[i-w2,j+w1]+integral_yz[i-w2,j-w1]
                tan_yz_d = integral_yz[i+w2,j+w1]-integral_yz[i,j+w1]-integral_yz[i+w2,j-w1]+integral_yz[i,j-w1]
                tan_xx = (tan_xx_r-tan_xx_l)/((2*w1+1)*(2*w2+1))
                tan_xy = (tan_xy_r-tan_xy_l)/((2*w1+1)*(2*w2+1))
                tan_xz = (tan_xz_r-tan_xz_l)/((2*w1+1)*(2*w2+1))
                tan_yx = (tan_yx_d-tan_yx_u)/((2*w1+1)*(2*w2+1))
                tan_yy = (tan_yy_d-tan_yy_u)/((2*w1+1)*(2*w2+1))
                tan_yz = (tan_yz_d-tan_yz_u)/((2*w1+1)*(2*w2+1))
                tan_x = np.array([tan_xx,tan_xy,tan_xz])
                tan_y = np.array([tan_yx,tan_yy,tan_yz])
                norm = np.cross(tan_x,tan_y)
                #if (norm[0]*norm[1]*norm[2]<0):
                #    norm = -norm
                leng = np.linalg.norm(norm)
                if (leng>0):
                    valid[i,j]=True
                    normals[:,i,j] = norm / leng * self.normals_length
        print('normals computed')
        self.valid = valid
        return normals

    def run_kmeans(self):
        label_im = self.label

        all_in = np.zeros((self.H*self.W,9))
        all_in[:,0:3] = self.__mat2tab(color.rgb2lab(self.image))
        all_in[:,3:6] = self.__mat2tab(self.normals[self.current_index])
        all_in[:,6:9] = self.__mat2tab(self.point_cloud)
        
        whitened = whiten(all_in)

        groups = {}
        n_labels = 8
        centroidc, labelc = kmeans2(all_in[:,0:3],n_labels,iter=10, thresh=1e-05, minit='random', missing='warn', check_finite=True)
        groups.update({'color': {'centroid' : centroidc, 'label' : labelc}})
        centroidn, labeln = kmeans2(all_in[:,3:6],n_labels)
        groups.update({'normals': {'centroid' : centroidn, 'label' : labeln}})
        centroidp, labelp = kmeans2(all_in[:,6:9],n_labels)
        groups.update({'xyz': {'centroid' : centroidp, 'label' : labelp}})
        centroid, label = kmeans2(all_in[:,0:6],n_labels)
        groups.update({'normcolor': {'centroid' : centroid, 'label' : label}})
        centroid, label = kmeans2(all_in,8)
        groups.update({'all': {'centroid' : centroid, 'label' : label}})
        print('k-means done')
        return groups

    def __mat2tab(self,mat):
        if mat.shape[0]==3:
            n_rows =  int(mat.size/3)
            tab = np.zeros((n_rows,3))
            tab[:,0] = np.reshape(mat[0,:,:],n_rows)
            tab[:,1] = np.reshape(mat[1,:,:],n_rows)
            tab[:,2] = np.reshape(mat[2,:,:],n_rows)
            return tab

        if mat.shape.__len__()==3:
            n_rows =  int(mat.size/3)
            tab = np.zeros((n_rows,3))
            tab[:,0] = np.reshape(mat[:,:,0],n_rows)
            tab[:,1] = np.reshape(mat[:,:,1],n_rows)
            tab[:,2] = np.reshape(mat[:,:,2],n_rows)
            return tab
                    

    def __load_mat(self):
        try:
            self.dataset_dict = sio.loadmat(self.mat_path)
            self.n = self.dataset_dict['depths'][1,1,:].size
            return True
        except:
            pass
        return False

    def __depth2xyz(self):
        point_cloud = np.zeros((3, self.H, self.W))
        depth = self.depth
        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02
        [xx, yy] = np.meshgrid(range(1,self.W+1),range(1,self.H+1))
        point_cloud[0,:,:] = (xx - cx_rgb) * depth / fx_rgb
        y = (yy - cy_rgb) * depth / fy_rgb
        y = -y+np.ndarray.max(y)
        point_cloud[1,:,:] = depth
        point_cloud[2,:,:] = y
        return point_cloud



if __name__ == "__main__":
    #pc = plane_clustering(mat_path='../../3DFinalProj/mat/reduced_dataset.mat')
    pc = plane_clustering(mat_path='../reduced_reduced_dataset.mat',smoothed=True, sigma=0.8)
    pc.show()
    input("Press Enter to continue...")
    while (pc.next()): pass
    
    #print(pc.point_cloud)
    #plt.imshow(pc.image, interpolation='nearest')
    #plt.show()
    #plt.imshow(sndim.gaussian_filter(pc.image,sigma=(1,1,0),order=0))
    #plt.show()
