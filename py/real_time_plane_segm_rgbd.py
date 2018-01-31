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
import time

class plane_clustering:
    def __init__(self, **kwargs):
        self.current_index = -1
        self.normals = []
        self.clusters = []
        self.clustered_normals = []
        self.mat_path = kwargs.get('mat_path', '.')
        self.smoothed = kwargs.get('smoothed', False)
        self.sigm = kwargs.get('sigma', 1)
        self.n = 0
        if not self.__load_mat(): # self.dataset_dict, self.n
            print('Error loading \'mat\' dataset')
            exit()
        self.var_names = ('accelData', 'depths', 'rawDepths', 'images', 'instances',
                          'labels', 'names', 'sceneTypes', 'scenes')
        self.normals_length = 1
        for i in range(0,self.n):
            self.normals.append([])
            self.clusters.append([])
        # print((self.dataset_dict['names'][1][0][0]))
        # print(str(self.dataset_dict['names']))

        # computing average normals
        self.avg_win_long = 2  # half of the greatest dimension of the avg window
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
        self.normals.insert(self.current_index,self.compute_normals())
        self.clustered_normals.insert(self.current_index,self.cluster_normals())


        pc_tab = self.__mat2tab(self.point_cloud)
        floors_idx = self.find_floor()
        # pc_floors = pc_tab[floors_idx,:]

        
        planes = self.unsup_cluster(floors_idx)
        min_plane_dist = 0.01
        # pc_dist: {'01':d01, '02': d02, ... }
        # mean_dist =  [ [0,            mean(d_01), mean(d_02), mean(d_03), ... ],
        #                [mean(d_10),   0,          mean(d_12), mean(d_13), ... ],
        #                [ ...                                                  ] ]
        # dev_dist  =  [ [0,             std(d_01),  std(d_02),  std(d_03), ... ],
        #                [ std(d_10),   0,           std(d_12),  std(d_13), ... ],
        #                [ ...                                                  ] ]
        pc_dist, mean_dist, dev_dist = self.plane_distances(planes)
        
        self.merge_planes(planes,pc_dist,mean_dist,dev_dist)

        #print(str(planes))
        # quantization of the 3 components of the normals
        # mapping 3D normals into phi,theta space (n_x / n_z, n_y / n_z)
        # clustering
        # evaluation:
        # - take each custer
        # - take the most frequent label
        # - see the ratio between the number of points with the most frequent label and all the other points
        # - compare different clustering methods
        plt.imshow(self.image, interpolation='nearest')
        self.plot_pc(floors_idx)
        for i,p in enumerate(planes):
            # print('mean dist: {:.5f}'.format(p['dist']))
            idx =p['idx']
            if idx.__len__()>0:
                p = self.plot_pc(idx)
            

        return True

    def point_plane_dist(self,points_idx,plane):
        pc_tab = self.__mat2tab(self.point_cloud)
        le = points_idx.__len__()
        pc_plane = np.reshape(pc_tab[points_idx,:],(le,3))
        PC_plane = np.hstack((pc_plane,np.ones((le,1))))
        tmp = np.dot(PC_plane,np.reshape(plane,(4,1)))
        distances = [abs(tmp[i])/np.sqrt(plane[0]**2+plane[1]**2+plane[2]**2) for i in range(le)]
        return distances

    def plane_distances(self,planes):
        pc_dist = {}
        mean_dist = np.zeros((planes.__len__(),planes.__len__()))
        dev_dist = np.zeros((planes.__len__(),planes.__len__()))
        for i,p1 in enumerate(planes):
            idx = p1['idx']
            le = idx.__len__()
            for j,p2 in enumerate(planes):
                plane_coeff = p2['eq']
                distances = self.point_plane_dist(idx,plane_coeff)
                # sq_distances = [distances[i]**2 for i in range(distances.__len__())]
                pc_dist[str(i)+str(j)] = distances
                mean_dist[i,j] = np.mean(distances)
                dev_dist[i,j] = np.std(distances)
                # print('d['+str(i)+','+str(j)+']: {:2.2E} +- {:2.2E}'.format(mean_dist[i,j],dev_dist[i,j]))
        return pc_dist, mean_dist, dev_dist

    def unsup_cluster(self,data_idx):
        # http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
        max_mean_dist = 0.001
        max_sum_xy = 0.2
        min_n_p = round(data_idx.__len__()/50)
        pc_tab = self.__mat2tab(self.point_cloud)
        data = pc_tab[data_idx,:]
        import sklearn.mixture as skm
        bgm = skm.BayesianGaussianMixture(n_components=50,covariance_type='full',tol=1e-3,reg_covar=1e-6, 
        max_iter=200,n_init=2,init_params='kmeans',weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=None,mean_precision_prior=None,mean_prior=None,degrees_of_freedom_prior=None,
        covariance_prior=None,random_state=None,warm_start=False,verbose=0,verbose_interval=10)
        #bgm.fit(floors, self.__mat2tab(self.label))
        bgm.fit(data)
        predicted = bgm.predict(data)
        planes = []
        for k in range(np.min(predicted),np.max(predicted)+1):
            klabel = [i for i,j in enumerate(predicted) if j==k]
            if (klabel.__len__()>10):
                # print('n. of points: '+str(klabel.__len__()))
                idx = [data_idx[kk] for kk in klabel]
                le = idx.__len__()
                plane_coeff = self.__find_plane(pc_tab[idx,:])
                distances = self.point_plane_dist(idx,plane_coeff)
                sq_distances = [distances[i]**2 for i in range(distances.__len__())]
                mean_dist = np.mean(sq_distances)
                if (mean_dist<max_mean_dist and le>min_n_p and abs(plane_coeff[0])+abs(plane_coeff[1])<max_sum_xy):
                    planes.append({'dist':mean_dist,'eq':plane_coeff,'idx':idx})
                    print('[a,b,c,d]=[{0[0]:.2f},{0[1]:.2f},{0[2]:.2f},{0[3]:.2f}]; '.format(plane_coeff)+str(le)+' points; '+'dist: {:2.2E}'.format(mean_dist))
        return planes

    def __merge(self,planes,tab,pc_dist,mean_dist):
        conf = 0.01
        for j in range(tab.shape[1]):
            a = [k for k in range(tab.shape[0]) if tab[k,j]==True]
            p2 = planes[j]
            idx2 = p2['idx']
            if a.__len__()>0:
                print('merging '+str(a) + ' to '+str(j))
            for ind in range(a.__len__()):
                i = a[ind]
                p1 = planes[i]
                idx1 = p1['idx']
                le1 = idx1.__len__()
                distances = pc_dist[str(i)+str(j)]
                # print('dist_len = {:d}, idx1_len = {:d}, idx2_len = {:d}'.format(distances.__len__(),idx1.__len__(),idx2.__len__()))
                to_be_merged = [idx1[k] for k in range(le1) if distances[k]<=mean_dist[i,j]+conf]
                if (to_be_merged.__len__()>0):
                    p2['idx'] = np.hstack((idx2,to_be_merged))
                    
                    print('merging '+str(i)+'->'+str(j)+'. d={:.2e}; {:d}/{:d} points added'.format(mean_dist[i,j],to_be_merged.__len__(),le1))
            # re computing distances
            for ii in range(planes.__len__()):
                if not ii==j:
                    pc_dist[str(j)+str(ii)] = self.point_plane_dist(p2['idx'],planes[ii]['eq'])
        # to_be_deleted = 0
        # to_be_deleted_ind = []
        # for i in range(tab.shape[0]):
        #     tmp = [k for k in range(tab.shape[1]) if tab[i,k]==True]
        #     if tmp.__len__()>0:
        #         to_be_deleted = to_be_deleted + 1
        #         to_be_deleted_ind.append(i)
        
        # n_el = [planes[i].__len__() for i in range(planes.__len__())]
        # ord_n_el = np.sort(n_el)
        # sum_el = np.cumsum(ord_n_el)
        # sum_el = [sum_el[i]/sum_el[sum_el.__len__()-1] for i in range(sum_el.__len__())]
        
        # indexes = np.argsort(n_el)
        # to_keep = [sum_el[i] for i in range(sum_el.__len__()) if sum_el[i]>0.15]
        # to_be_deleted = n_el.__len__()-to_keep.__len__()
        # indexes = [indexes[i] for i in range(to_be_deleted)]

        # deleted = 0
        # for i in range(indexes.__len__()):
        #     if tmp.__len__()>0:
        #         planes.__delitem__(indexes[i]-deleted)
        #         deleted = deleted + 1            

    def merge_planes(self,planes,pc_dist,mean_dist,dev_dist):
        plane_union_threshold = 0.05
        tab = np.tile(np.array(False), [planes.__len__(),planes.__len__()])
        for i,p1 in enumerate(planes):
            plane_coeff = p1['eq']
            d1 = abs(p1['eq'][3])/np.sqrt(p1['eq'][0]**2+p1['eq'][1]**2+p1['eq'][2]**2)
            p1['d_from_origin'] = d1
            for j,p2 in enumerate(planes):
                if (j>i):
                    c1 = mean_dist[i,j]+dev_dist[i,j]-plane_union_threshold
                    c2 = mean_dist[j,i]+dev_dist[j,i]-plane_union_threshold
                    if (c1<0) or (c2<0):
                        if (c1*c2<0): # only one condition is true
                            if (c1<0): # points of p1 are close enough to p2
                                tab[i,j] = True
                            else: # points of p2 are close enough to p1
                                tab[j,i] = True
                        else:
                            if (mean_dist[i,j]<mean_dist[j,i]):
                                tab[i,j] = True
                            else: # points of p2 are close enough to p1
                                tab[j,i] = True
        self.__merge(planes,tab,pc_dist,mean_dist)

    def find_floor(self):
        # n = self.normals[self.current_index]
        # n_tab = self.__mat2tab(n)
        nx = self.clustered_normals[self.current_index]['x']
        ny = self.clustered_normals[self.current_index]['y']
        nz = self.clustered_normals[self.current_index]['z']
        phi = self.clustered_normals[self.current_index]['phi']
        theta = self.clustered_normals[self.current_index]['theta']
        # n_tab = np.hstack((np.reshape(nx,(nx.__len__(),1)),np.reshape(ny,(ny.__len__(),1)),np.reshape(nz,(nz.__len__(),1))))
        c1 = [abs(abs(nz[i])-1) for i in range(nz.__len__())]
        c3 = [abs(abs(phi[i])-90) for i in range(nz.__len__())]
        c5 = [abs(abs(phi[i])-270) for i in range(nz.__len__())]

        t = 1.05
        threshold1 = min(c1)*t
        threshold3 = min(c3)*t
        threshold5 = min(c5)*t
        
        print('t1:{:.2f},t3:{:.2f},t5:{:.2f}'.format(threshold1[0],threshold3[0], threshold5[0]))
        indexes = []
        cont_ang = 0
        cont_lin = 0
        for i in range(0,self.W*self.H):
            if c3[i] <= threshold3 or c5[i] <= threshold5:
                # indexes.append(i)
                cont_ang = cont_ang + 1
                if c1[i] <= threshold1:
                    indexes.append(i)
                    cont_lin = cont_lin + 1
        print('ang:'+str(cont_ang)+'\nlin:'+str(cont_lin))
        return indexes


    def plot_pc(self,plane_idx):
        point_size = 0.1
        line_width = 0.4
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        pc_tab = self.__mat2tab(self.point_cloud)
        norm_tab = self.__mat2tab(self.normals[self.current_index])
        ax.scatter(pc_tab[plane_idx, 0], pc_tab[plane_idx, 1], pc_tab[plane_idx, 2], 
        c='b',s=point_size)
        ax.azim = -90
        ax.elev = 25
        ax.set_xlim([self.xmin,self.xmax])
        ax.set_ylim([self.ymin,self.ymax])
        ax.set_zlim([0,self.zmax])
        # q = ax.quiver(pc_tab[plane_idx, 0], pc_tab[plane_idx, 1], pc_tab[plane_idx, 2],
        # norm_tab[plane_idx, 0],norm_tab[plane_idx, 1],norm_tab[plane_idx, 2], lw=line_width)
        plt.draw()
        plt.show()
        return {'ax':ax, 'fig': fig, 'plt':plt}

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
                left = self.point_cloud[:,i,j-1]
                right = self.point_cloud[:,i,j]
                up = self.point_cloud[:,i-1,j-1]
                down = self.point_cloud[:,i+1,j]
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
        max_w = np.maximum(w1,w2)
        for i in range(max_w,H-max_w):
            for j in range(max_w,W-max_w):
                tan_xx = (integral_xx[i+w2,j+w1]-integral_xx[i-w2-1,j+w1]-integral_xx[i+w2,j-w1-1]
                +integral_xx[i-w2-1,j-w1-1])/((2*w1+1)*(2*w2+1))
                tan_xy = (integral_xy[i+w2,j+w1]-integral_xy[i-w2-1,j+w1]-integral_xy[i+w2,j-w1-1]
                +integral_xy[i-w2-1,j-w1-1])/((2*w1+1)*(2*w2+1))
                tan_xz = (integral_xz[i+w2,j+w1]-integral_xz[i-w2-1,j+w1]-integral_xz[i+w2,j-w1-1]
                +integral_xz[i-w2-1,j-w1-1])/((2*w1+1)*(2*w2+1))
                tan_yx = (integral_yx[i+w1,j+w2]-integral_yx[i-w1-1,j+w2]-integral_yx[i+w1,j-w2-1]
                +integral_yx[i-w1-1,j-w2-1])/((2*w1+1)*(2*w2+1))
                tan_yy = (integral_yy[i+w1,j+w2]-integral_yy[i-w1-1,j+w2]-integral_yy[i+w1,j-w2-1]
                +integral_yy[i-w1-1,j-w2-1])/((2*w1+1)*(2*w2+1))
                tan_yz = (integral_yz[i+w1,j+w2]-integral_yz[i-w1-1,j+w2]-integral_yz[i+w1,j-w2-1]
                +integral_yz[i-w1-1,j-w2-1])/((2*w1+1)*(2*w2+1))
                tan_x = np.array([tan_xx,tan_xy,tan_xz])
                tan_y = np.array([tan_yx,tan_yy,tan_yz])
                norm = np.cross(tan_x,tan_y)
                # if (norm[0]*norm[1]*norm[2]<0):
                #    norm = -norm
                leng = np.linalg.norm(norm)
                if (leng>0):
                    valid[i,j]=True
                    normals[:,i,j] = norm / leng
        print('normals computed')
        self.valid = valid
        return normals #self.normals[self.current_index]
    
    def quantize(self,vec,nbins):
        # print('vec in [{:.1f},{:.1f}]'.format(min(vec)[0],max(vec)[0]))
        nbins = round(nbins)
        vec = np.reshape(vec,(vec.__len__(),))
        sorted_ind = np.argsort(vec)
        sorted_vec = np.sort(vec)
        res = np.zeros((vec.__len__(),1))
        # values, base = np.histogram(sorted_vec, bins='rice')
        values, base = np.histogram(sorted_vec, bins=nbins)
        centers = [(base[i+1]+base[i])/2 for i in range(base.__len__()-1)]
        cumulative = np.cumsum(values)
        cumulative = [float(cumulative[i])/float(cumulative[cumulative.__len__()-1]) for i in range(cumulative.__len__())]
        cumulative.reverse()
        cumulative.append(0)
        cumulative.reverse()
        d_cumulative =  [(cumulative[i+1] - cumulative[i])/ (base[i+1]-base[i]) for i in range(cumulative.__len__()-1)]
        cumulative.remove(0)
        threshold = 1/(base[base.__len__()-1]-base[0])
        # for i in range(d_cumulative.__len__()):
        #     print("[{:.2f},{:.2f}], c:{:.3f}, cum:{:.3f}, d_cum:{:.3f}".format(base[i],base[i+1],centers[i],cumulative[i],d_cumulative[i]))
        slow_growth_ind = [i for i,d in enumerate(d_cumulative) if d<threshold]
        fast_growth_ind = [i for i,d in enumerate(d_cumulative) if d>=threshold]
        # print('sgi:'+str(slow_growth_ind))
        # print('fgi:'+str(fast_growth_ind))
        s_from = 0
        s_to = slow_growth_ind.__len__()-1
        while s_from < slow_growth_ind.__len__():
            s_to = s_from
            while s_to+1<slow_growth_ind.__len__() and slow_growth_ind[s_to+1]<base.__len__() and slow_growth_ind[s_to+1]==slow_growth_ind[s_to]+1: s_to=s_to+1
            inizio = base[slow_growth_ind[s_from]]
            fine = base[slow_growth_ind[s_to]+1]
            avg = (inizio + fine)/2.0
            # print( 'sg:{:2.2f} -> {:.2f} => {:.2f}'.format(inizio,fine,avg))
            for ind, el in enumerate(sorted_vec):
                if el <= fine and el >= inizio:
                    sorted_vec[ind] = avg

            s_from = s_to+1
        for i, fgi in enumerate(fast_growth_ind):
            inizio = base[fgi]
            fine = base[fgi+1]
            avg = (inizio + fine)/2.0
            # print( 'fg:{:2.2f} -> {:.2f} => {:.2f}'.format(inizio,fine,avg))
            for ind, el in enumerate(sorted_vec):
                    if el <= fine and el >= inizio:
                        sorted_vec[ind] = avg
        for i,si in enumerate(sorted_ind):
            res[si,0] = sorted_vec[i]
        return res

    
    def cluster_normals(self):
        from math import atan, pi

        nx = self.__mat2tab(self.normals[self.current_index][0,:,:])
        ny = self.__mat2tab(self.normals[self.current_index][1,:,:])
        nz = self.__mat2tab(self.normals[self.current_index][2,:,:])
        phi = np.zeros((nx.shape[0],1))
        theta = np.zeros((nx.shape[0],1))
        
        for i in range(nx.__len__()):
            if nx[i]>0:
                phi[i] = atan(nz[i]/nx[i])*180/pi+180*(nx[i]<0)
                phi[i] = phi[i] +360*(phi[i]<0)
            if ny[i]>0:
                theta[i] = atan(nz[i]/ny[i])*180/pi+180*(ny[i]<0)
                theta[i] = theta[i] +360*(theta[i]<0)
        # print('phi in [{:.3f},{:.3f}], theta in [{:.3f},{:.3f}]'.format(min(phi)[0],max(phi)[0],min(theta)[0],max(theta)[0]))
        levels_lin = 32
        levels_deg = 36
        qnx = self.quantize(nx,levels_lin)
        qny = self.quantize(ny,levels_lin)
        qnz = self.quantize(nz,levels_lin)
        qphi = self.quantize(phi,levels_deg)
        qtheta = self.quantize(theta,levels_deg)
        return {'x': qnx, 'y': qny, 'z': qnz, 'phi':qphi, 'theta': theta}

    def run_kmeans(self):
        label_im = self.label
        all_in = np.zeros((self.H*self.W,9))
        all_in[:,0:3] = self.__mat2tab(color.rgb2lab(self.image))
        all_in[:,3:6] = self.__mat2tab(self.normals[self.current_index])
        all_in[:,6:9] = self.__mat2tab(self.point_cloud)
        whitened = whiten(all_in)
        groups = {}
        n_labels = self.n_labels
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
        return np.reshape(mat,(mat.size,1))

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
        self.xmin = np.ndarray.min(point_cloud[0,:,:])
        self.xmax = np.ndarray.max(point_cloud[0,:,:])
        self.ymin = np.ndarray.min(point_cloud[1,:,:])
        self.ymax = np.ndarray.max(point_cloud[1,:,:])
        self.zmax = np.ndarray.max(y)
        return point_cloud

    def __find_plane(self,points):
        # import time
        Points = np.hstack((points,np.ones((points.shape[0],1))))
        # time_start = time.clock()
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.transpose(Points),Points))
        # time_elapsed = (time.clock() - time_start)
        # print(time_elapsed)
        # time_start = time.clock()
        # u, s, vh = np.linalg.svd(Points, full_matrices=True)
        # time_elapsed = (time.clock() - time_start)
        # print(time_elapsed)
        plane_coeff = -eigenvectors[:,3]/np.linalg.norm(eigenvectors[:,3])
        # plane_coeff_2 = vh[3,:]
        # print('svd:\n'+str(str(vh)))
        # print('eigenvectors:\n'+str(eigenvectors))
        # print('a,b,c,d (eig):\n'+str(plane_coeff))
        # print('a,b,c,d (svd):\n'+str(plane_coeff_2))
        return plane_coeff

if __name__ == "__main__":
    #pc = plane_clustering(mat_path='../reduced_reduced_dataset.mat')
    pc = plane_clustering(mat_path='../reduced_reduced_dataset.mat',smoothed=True, sigma=0.8)
    pc.show()
    input("Press Enter to continue...")
    while (pc.next()): input("Press Enter to continue...")
    
    #print(pc.point_cloud)
    #plt.imshow(pc.image, interpolation='nearest')
    #plt.show()
    #plt.imshow(sndim.gaussian_filter(pc.image,sigma=(1,1,0),order=0))
    #plt.show()
