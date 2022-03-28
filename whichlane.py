import numpy as np
import torch
import  cv2
# 得到车子的中心点， 自车的中心点 可以使用 （x=0， z=1,y=0.5）

img=cv2.imread('/home/l/code_for_vscode/decision_making_code/nuscenes1.jpg')
print(img)

cars=np.random.rand(100,3)
ego=[[0.0,0.5,2.0]]      # 第一个数字表示向右， 第二个表示向下， 第三个表示向前
ego=np.array(ego)
cars_and_ego=np.concatenate((cars, ego))
cars_and_ego=ego
cars_and_ego=torch.from_numpy(cars_and_ego)


# 得到相机的内参矩阵 前置相机的内参矩阵
neican=[
[
1266.417203046554,
0.0,
816.2670197447984
],
[
0.0,
1266.417203046554,
491.50706579294757
],
[
0.0,
0.0,
1.0
]
]
neican=np.array(neican)
neican=torch.from_numpy(neican)


# 根据车子的中心点和内参矩阵 计算像素中的位置。
def points_cam2img(points_3d, proj_mat, with_depth=False):

    """相机坐标系到像素坐标系的投影

    Args:
        points_3d (torch.Tensor | np.ndarray): Points in shape (N, 3)
        proj_mat (torch.Tensor | np.ndarray): 内参矩阵
            Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        (torch.Tensor | np.ndarray): Points in image coordinates,
            with shape [N, 2] if `with_depth=False`, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res


locatations_in_image=points_cam2img(cars_and_ego, neican,1)

# 判断每个车在哪个车道上：
for xy in locatations_in_image:
    pass
    # lane_x :{locationxy:[], type:dashed, uncertainty:xx} #



point_size = 1
point_color = (0, 0, 255) # BGR
thickness = 40 #  0 、4、

print(locatations_in_image[-1])
cv2.circle(img, (int(locatations_in_image[-1][0]),int(locatations_in_image[-1][1])), point_size, point_color, thickness)

cv2.imshow('imadge', img)
cv2.waitKey(0)
#cv2.destroyAllWindows()
