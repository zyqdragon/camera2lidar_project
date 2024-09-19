import numpy as np
import cv2
import math
import open3d as o3d
import time

##############################################################################################
object_3d_points = np.array(([6.131, -2.885,  0.222],
                             [6.224, -4.143, -0.903],
                             [6.986, -0.918,  0.54 ],
                             [7.277,  0.381, -0.278],
                             [8.098,  1.310, -0.18 ],
                             [9.816, -0.511,  1.256],                             
                             [8.181,  2.736, -0.604],
                             [8.107,  1.21 , -0.562]), dtype=np.double)
# object_2d_point为与三维点云坐标对应的图像中的二维坐标，通过画图软件及点中后左下角的坐标确定。
object_2d_point = np.array(([int(1095)  , int(153)],
                            [int(1210)  , int(317)],
                            [int(808)  , int(87)],
                            [int(622)  , int(198)],
                            [int(509)  , int(185)],
                            [int(723)  , int(37)],
                            [int(345)  , int(240)],
                            [int(527)  , int(231)]), dtype=np.double)

camera_matrix = np.array(([1083.4     ,    0. ,        604.58],
                          [  0.       ,  1075 ,        337.41],
                          [  0.       ,    0.      ,     1.        ]), dtype=np.double)
dist_coefs = np.array([ -0.5702,  0.2705, -0.0014,   0.0040,  0.0000], dtype=np.double)# 求解相机位姿
############################################################################
found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs,flags=cv2.SOLVEPNP_SQPNP)
rotM = cv2.Rodrigues(rvec)[0]
print("----rotM=",rotM)
print("----tvec=",tvec)
camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
print("---------camera_postion.T=",camera_postion.T)
###################################### EPNP ##############################################################
imgpts, jac = cv2.projectPoints(np.array([6.131, -2.885,  0.222], dtype=np.float32), rvec, tvec, camera_matrix, dist_coefs)
print("-----imgpts=",imgpts)
imgpts, jac = cv2.projectPoints(np.array([6.224, -4.143, -0.903], dtype=np.float32), rvec, tvec, camera_matrix, dist_coefs)
print("-----imgpts=",imgpts)
imgpts, jac = cv2.projectPoints(np.array([6.986, -0.918,  0.54], dtype=np.float32), rvec, tvec, camera_matrix, dist_coefs)
print("-----imgpts=",imgpts)
imgpts, jac = cv2.projectPoints(np.array([7.277,  0.381, -0.278], dtype=np.float32), rvec, tvec, camera_matrix, dist_coefs)
print("-----imgpts=",imgpts)
imgpts, jac = cv2.projectPoints(np.array([8.098,  1.310, -0.18 ], dtype=np.float32), rvec, tvec, camera_matrix, dist_coefs)
print("-----imgpts=",imgpts)
imgpts, jac = cv2.projectPoints(np.array([9.816, -0.511,  1.256], dtype=np.float32), rvec, tvec, camera_matrix, dist_coefs)
print("-----imgpts=",imgpts)
imgpts, jac = cv2.projectPoints(np.array([8.181,  2.736, -0.604], dtype=np.float32), rvec, tvec, camera_matrix, dist_coefs)
print("-----imgpts=",imgpts)
imgpts, jac = cv2.projectPoints(np.array([8.107,  1.21 , -0.562], dtype=np.float32), rvec, tvec, camera_matrix, dist_coefs)
print("-----imgpts=",imgpts)
######################################################################################################################
###########################image to pcd######################################################
image_3d_points = np.array(([605, 253,  1],
                            [825, 67, 1]), dtype=np.double)
# image_3d_points = np.array(([605, 253,  1]), dtype=np.double)

camera_matrix_inv=np.linalg.inv(camera_matrix)
image_temp=camera_matrix_inv@(image_3d_points.T)
print("-----image_temp=",image_temp)
# print("--result=",camera_matrix_inv@camera_matrix)

print("-----------image_temp[1][0]=",image_temp[1][0])


Out_matrix = np.concatenate((rotM, tvec), axis=1)

print("--------Out_matrix=",Out_matrix)

pointcloud3 = o3d.io.read_point_cloud("whole_pcd_down.pcd")
image2=cv2.imread('3.jpg')
pcd_points=np.array(pointcloud3.points)
print("---------len_pcd_points=",len(pcd_points))

t1=time.time()
object_point=np.array([[0,0,0]])

for idx in range(80000):
    pcd_extend=np.array([pcd_points[idx][0],pcd_points[idx][1],pcd_points[idx][2],1])
    pcd_temp=Out_matrix@pcd_extend
    pcd_temp2=pcd_temp/pcd_temp[2]
    if (pcd_temp2[0]>image_temp[0][0]) and (pcd_temp2[1]<image_temp[1][0]) and (pcd_temp2[0]<image_temp[0][1]) and (pcd_temp2[1]>image_temp[1][1]):
        # print("---------pcd_temp2=",pcd_temp2)
        imgpts, jac = cv2.projectPoints(pcd_points[idx], rvec, tvec, camera_matrix, dist_coefs)
        # earth_points_w=np.append(earth_points_w,np.array([[idx*step+left_upper_corner[0],left_upper_corner[1],z_value]]), axis=0)
        # object_point=np.append(object_point,object_point,axis=0)
        object_point=np.append(object_point,np.array([[pcd_points[idx][0],pcd_points[idx][1],pcd_points[idx][2]]]),axis=0)
        # print("-----imgpts=",imgpts)
        # radius=np.sqrt(np.square(pcd_points[idx][0])+np.square(pcd_points[idx][1])+np.square(pcd_points[idx][2]))
        # cv2.circle(image2, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), 2, (int(radius/20*255), int(radius/12*255), 255-int(radius/12*255)), 2)
        cv2.circle(image2, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), 2, (180, 199, 0), 2)
object_point=object_point[1:]
object_pcd=o3d.geometry.PointCloud()   # original point clouds
object_pcd.points=o3d.utility.Vector3dVector(object_point)
o3d.io.write_point_cloud("object_pcd.pcd", object_pcd)
print("-----------time required for 30000 points is:",time.time()-t1)
print("-----------object_point=",object_point)
cv2.imshow("image2pcd_result.png", image2) 

##########################################################################################
def radius_outlier_removal(pcd, radius=0.80, min_points=990):  # 0.8m stone
    cl, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
    pcd = pcd.select_by_index(ind)
    return pcd

# 示例
# pcd = o3d.io.read_point_cloud("object_pcd.pcd")
pcd = radius_outlier_removal(object_pcd)
o3d.visualization.draw_geometries([pcd])
# pcd = o3d.io.read_point_cloud("object_pcd.pcd")
print(pcd)  # 输出点云点的个数
obb = pcd.get_oriented_bounding_box()
obb.color = (0, 1, 0)  # obb包围盒为绿色
 
[center_x, center_y, center_z] = obb.get_center()
print("obb包围盒的中心坐标为：\n", [center_x, center_y, center_z])
 
vertex_set = np.asarray(obb.get_box_points())
print("obb包围盒的顶点为：\n", vertex_set)
 
max_bound = np.asarray(obb.get_max_bound())
print("obb包围盒边长的最大值为：\n", max_bound)
 
min_bound = np.asarray(obb.get_min_bound())
print("obb包围盒边长的最小值为：\n", min_bound)
 
o3d.visualization.draw_geometries([pcd, obb], window_name="OBB包围盒",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)
####################tongji filter####################################
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2)
# pcd = pcd.select_by_index(ind)
# # 显示过滤后的点云
# o3d.visualization.draw_geometries([pcd])
cv2.waitKey()