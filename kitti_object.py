""" Helper class and functions for loading KITTI objects

raw Author: Charles R. Qi
Date: September 2017

modified by Yao Li
Date: May 2020
"""
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
import linecache #读取指定行
import time
import matplotlib.pyplot as plt #仅用于延时，不知有用否

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mayavi"))
import kitti_util as utils
import argparse

trackdirid=['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009','0010','0011',
            '0012','0013','0014','0015','0016','0017','0018','0019','0020','0021','0022',
            '0023','0024','0025','0026','0027','0028']
timeintervel=300   # every 50ms play recommanded

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

cbox = np.array([[0, 70.4], [-40, 40], [-3, 1]])


class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="training", args=None):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split
        print(root_dir, split)
        self.split_dir = os.path.join(root_dir, split)

        if split == "training":
            #self.num_samples = 7481
            self.num_samples = 8008  #tracking
        elif split == "testing":
            self.num_samples = 7518
        else:
            print("Unknown split: %s" % (split))
            exit(-1)

        lidar_dir = "velodyne"
        depth_dir = "depth"
        pred_dir = "pred"
        if args is not None:
            lidar_dir = args.lidar
            depth_dir = args.depthdir
            pred_dir = args.preddir

        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.label_dir = os.path.join(self.split_dir, "label_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")

        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.depth_dir = os.path.join(self.split_dir, depth_dir)
        self.pred_dir = os.path.join(self.split_dir, pred_dir)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        #assert idx < self.num_samples #ly  下面的assert均为修改的
        #img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
        img_filename = os.path.join(self.image_dir, "%s.png" % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float64, n_vec=4):
        #assert idx < self.num_samples #ly
        #lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
        lidar_filename = os.path.join(self.lidar_dir, "%s.bin" % (idx))
        print(lidar_filename)
        return utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_calibration(self, idx):
        #assert idx < self.num_samples
        calib_filename = os.path.join(self.calib_dir, "%s.txt" % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        #assert idx < self.num_samples and self.split == "training"
        #label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        label_filename = os.path.join(self.label_dir, "%s.txt" % (idx))
        return utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        #assert idx < self.num_samples
        idx=idx.spilt('-')#ly
        pred_filename = os.path.join(self.pred_dir, "%s.txt" % (idx[0]))#ly
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None

    def get_depth(self, idx):
        #assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%s.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_image(self, idx):
        #assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%s.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_pc(self, idx):
        #assert idx < self.num_samples
        lidar_filename = os.path.join(self.depthpc_dir, "%s.bin" % (idx))
        is_exist = os.path.exists(lidar_filename)
        if is_exist:
            return utils.load_velo_scan(lidar_filename), is_exist
        else:
            return None, is_exist
        # print(lidar_filename, is_exist)
        # return utils.load_velo_scan(lidar_filename), is_exist

    def get_top_down(self, idx):
        pass

    def isexist_pred_objects(self, idx):
        #assert idx < self.num_samples and self.split == "training"
        pred_filename = os.path.join(self.pred_dir, "%s.txt" % (idx))
        return os.path.exists(pred_filename)

    def isexist_depth(self, idx):
        #assert idx < self.num_samples and self.split == "training"
        depth_filename = os.path.join(self.depth_dir, "%s.txt" % (idx))
        return os.path.exists(depth_filename)


class kitti_object_video(object):
    """ Load data for KITTI videos """

    def __init__(self, img_dir, lidar_dir, calib_dir):
        self.calib = utils.Calibration(calib_dir, from_video=True)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted(
            [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
        )
        self.lidar_filenames = sorted(
            [os.path.join(lidar_dir, filename) for filename in os.listdir(lidar_dir)]
        )
        print(len(self.img_filenames))
        print(len(self.lidar_filenames))
        # assert(len(self.img_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = self.img_filenames[idx]
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert idx < self.num_samples
        lidar_filename = self.lidar_filenames[idx]
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib


def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, "dataset/2011_09_26/")
    dataset = kitti_object_video(
        os.path.join(video_path, "2011_09_26_drive_0023_sync/image_02/data"),
        os.path.join(video_path, "2011_09_26_drive_0023_sync/velodyne_points/data"),
        video_path,
    )
    print(len(dataset))
    for i in range(len(dataset)):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        cv2.imshow("video", img)
        draw_lidar(pc)
        raw_input()
        pc[:, 0:3] = dataset.get_calibration().project_velo_to_rect(pc[:, 0:3])
        draw_lidar(pc)
        raw_input()
    return


def show_image_with_boxes(img, objects=None, calib=None, show3d=True, depth=None,track_pred=None):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 3d detection bbox
    img2 = np.copy(img)  # for 3d tracking bbox
    img3 = np.copy(img)  # for 3d bbox
    if objects is not None:
        for obj in objects:
            # if obj.type == "DontCare":
            #     continue
            # cv2.rectangle(
            #     img1,
            #     (int(obj.xmin), int(obj.ymin)),
            #     (int(obj.xmax), int(obj.ymax)),
            #     (0, 255, 0),
            #     2,
            # )
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            try:                     #判断读到的结果数据是否有问题
                box3d_pts_2d.any()
            except:                 # 不用抛出错误，有错跳出本次循环即可，加上错误类型时会抛出错误
                print('detection results hanve some problems!')
                continue
            else:
                img1 = utils.draw_projected_box3d(img1, box3d_pts_2d)

    if track_pred is not None:# tracking results
        for obj in track_pred:
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            #print(box3d_pts_2d)
            try:                     #判断读到的结果数据是否有问题
                box3d_pts_2d.any()
            except:                 # 不用抛出错误，有错跳出本次循环即可，加上错误类型时会抛出错误
                print('tracking results hanve some problems!')
                continue
            else:
                img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img2,'trackid:%d'%obj.trackid, (int((obj.xmin+obj.xmax)/2.1), int(obj.ymin-10)), font, 0.5, (0, 255, 0), 1)
                        #画trackid 绘制的文字，位置，字型，字体大小，文字颜色，线型




        # project
        # box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # box3d_pts_32d = utils.box3d_to_rgb_box00(box3d_pts_3d_velo)
        # box3d_pts_32d = calib.project_velo_to_image(box3d_pts_3d_velo)
        # img3 = utils.draw_projected_box3d(img3, box3d_pts_32d)
    # print("img1:", img1.shape)
    #cv2.imshow("2dbox", img1)

    cv2.namedWindow("tracking results", cv2.WINDOW_FREERATIO)  # 可鼠标调节自适应比例
    cv2.namedWindow("detection results", cv2.WINDOW_FREERATIO)  # 可鼠标调节自适应比例
    cv2.imshow("detection results", img1)
    # print("img3:",img3.shape)
    # Image.fromarray(img3).show()
    #show3d = True
    if show3d:# show tracking results
        # print("img2:",img2.shape)
        cv2.imshow("tracking results", img2)
    if depth is not None:
        cv2.imshow("depth", depth)


def show_image_with_boxes_3type(img, objects, calib, objects2d, name, objects_pred):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    type_list = ["Pedestrian", "Car", "Cyclist"]
    # draw Label
    color = (0, 255, 0)
    for obj in objects:
        if obj.type not in type_list:
            continue
        cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            color,
            3,
        )
    startx = 5
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lables = [obj.type for obj in objects if obj.type in type_list]
    text_lables.insert(0, "Label:")
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    # draw 2D Pred
    color = (0, 0, 255)
    for obj in objects2d:
        cv2.rectangle(
            img1,
            (int(obj.box2d[0]), int(obj.box2d[1])),
            (int(obj.box2d[2]), int(obj.box2d[3])),
            color,
            2,
        )
    startx = 85
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lables = [type_list[obj.typeid - 1] for obj in objects2d]
    text_lables.insert(0, "2D Pred:")
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    # draw 3D Pred
    if objects_pred is not None:
        color = (255, 0, 0)
        for obj in objects_pred:
            if obj.type not in type_list:
                continue
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                color,
                1,
            )
        startx = 165
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_lables = [obj.type for obj in objects_pred if obj.type in type_list]
        text_lables.insert(0, "3D Pred:")
        for n in range(len(text_lables)):
            text_pos = (startx, 25 * (n + 1))
            cv2.putText(
                img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA
            )

    cv2.imshow("with_bbox", img1)
    cv2.imwrite("imgs/" + str(name) + ".png", img1)


def get_lidar_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def get_lidar_index_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    return fov_inds


def depth_region_pt3d(depth, obj):
    b = obj.box2d
    # depth_region = depth[b[0]:b[2],b[2]:b[3],0]
    pt3d = []
    # import pdb; pdb.set_trace()
    for i in range(int(b[0]), int(b[2])):
        for j in range(int(b[1]), int(b[3])):
            pt3d.append([j, i, depth[j, i]])
    return np.array(pt3d)


def get_depth_pt3d(depth):
    pt3d = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            pt3d.append([i, j, depth[i, j]])
    return np.array(pt3d)


def show_lidar_with_depth(
    pc_velo,
    objects,
    calib,
    fig,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
    constraint_box=False,
    pc_label=False,
    save=False,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        pc_velo = pc_velo[pc_velo_index, :]
        print(("FOV point num: ", pc_velo.shape))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig, pc_label=pc_label)

    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))
        print("depth_pc_velo:", depth_pc_velo.shape)
        print("depth_pc_velo:", type(depth_pc_velo))
        print(depth_pc_velo[:5])
        draw_lidar(depth_pc_velo, fig=fig, pts_color=(1, 1, 1))

        if save:
            data_idx = 0
            vely_dir = "data/object/training/depth_pc"
            save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))
            print(save_filename)
            # np.save(save_filename+".npy", np.array(depth_pc_velo))
            depth_pc_velo = depth_pc_velo.astype(np.float32)
            depth_pc_velo.tofile(save_filename)

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color, label=obj.type)

    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    mlab.show(1)


def save_depth0(
    data_idx,
    pc_velo,
    calib,
    img_fov,
    img_width,
    img_height,
    depth,
    constraint_box=False,
):

    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        pc_velo = pc_velo[pc_velo_index, :]
        type = np.zeros((pc_velo.shape[0], 1))
        pc_velo = np.hstack((pc_velo, type))
        print(("FOV point num: ", pc_velo.shape))
    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))

        type = np.ones((depth_pc_velo.shape[0], 1))
        depth_pc_velo = np.hstack((depth_pc_velo, type))
        print("depth_pc_velo:", depth_pc_velo.shape)

        depth_pc = np.concatenate((pc_velo, depth_pc_velo), axis=0)
        print("depth_pc:", depth_pc.shape)

    vely_dir = "data/object/training/depth_pc"
    save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)


def save_depth(
    data_idx,
    pc_velo,
    calib,
    img_fov,
    img_width,
    img_height,
    depth,
    constraint_box=False,
):

    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc = np.hstack((depth_pc_velo, indensity))

        print("depth_pc:", depth_pc.shape)

    vely_dir = "data/object/training/depth_pc"
    save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)


def show_lidar_with_boxes(
    pc_velo,
    objects,
    calib,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    if img_fov:
        pc_velo = get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        print(("FOV point num: ", pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)
    # pc_velo=pc_velo[:,0:3]

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)

        # Draw depth
        if depth is not None:
            # import pdb; pdb.set_trace()
            depth_pt3d = depth_region_pt3d(depth, obj)
            depth_UVDepth = np.zeros_like(depth_pt3d)
            depth_UVDepth[:, 0] = depth_pt3d[:, 1]
            depth_UVDepth[:, 1] = depth_pt3d[:, 0]
            depth_UVDepth[:, 2] = depth_pt3d[:, 2]
            print("depth_pt3d:", depth_UVDepth)
            dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
            print("dep_pc_velo:", dep_pc_velo)

            draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))
        #

        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    mlab.show(1)


def box_min_max(box3d):
    box_min = np.min(box3d, axis=0)
    box_max = np.max(box3d, axis=0)
    return box_min, box_max


def get_velo_whl(box3d, pc):
    bmin, bmax = box_min_max(box3d)
    ind = np.where(
        (pc[:, 0] >= bmin[0])
        & (pc[:, 0] <= bmax[0])
        & (pc[:, 1] >= bmin[1])
        & (pc[:, 1] <= bmax[1])
        & (pc[:, 2] >= bmin[2])
        & (pc[:, 2] <= bmax[2])
    )[0]
    # print(pc[ind,:])
    if len(ind) > 0:
        vmin, vmax = box_min_max(pc[ind, :])
        return vmax - vmin
    else:
        return 0, 0, 0, 0


def stat_lidar_with_boxes(pc_velo, objects, calib):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """

    # print(('All point num: ', pc_velo.shape[0]))

    # draw_lidar(pc_velo, fig=fig)
    # color=(0,1,0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        v_l, v_w, v_h, _ = get_velo_whl(box3d_pts_3d_velo, pc_velo)
        print("%.4f %.4f %.4f %s" % (v_w, v_h, v_l, obj.type))


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    """ Project LiDAR points to image """
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(
            img,
            (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
            2,
            color=tuple(color),
            thickness=-1,
        )
    cv2.imshow("projection", img)
    return img


def show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None,track_pred=None):
    """ top_view image"""
    # print('pc_velo shape: ',pc_velo.shape)
    top_view = utils.lidar_to_top(pc_velo)
    top_image = utils.draw_top_image(top_view)
    print("top_image:", top_image.shape)

    # gt

    def bbox3d(obj):
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    # print("box2d BV:",boxes3d)
    lines = [obj.type for obj in objects if obj.type != "DontCare"]
    top_image = utils.draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    #pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]

        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )
    #tracking result
    if track_pred is not None:
        boxes3d = [bbox3d(obj) for obj in track_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in track_pred if obj.type != "DontCare"]
        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False,is_track=True
        )

    #cv2.namedWindow("top_image_bev", cv2.WINDOW_NORMAL)#窗口大小可变，左是窗口名称
    cv2.namedWindow("top_image_bev", cv2.WINDOW_FREERATIO)#可鼠标调节自适应比例
    #cv2.resizeWindow("top_image_bev", 640, 480)#ly
    cv2.imshow("top_image_bev", top_image)

def dataset_viz(root_dir, args):
    dataset = kitti_object(root_dir, split=args.split, args=args)
    ## load 2d detection results    or 3D results
    #objects2ds = read_det_file("box2d.list")
    imagedir = '0000'  # imagedir 第几组 default 0000
    argdetectdir = 'car_sassd_det_val' #which detection result file name,default saresult_val
    beginid={0000:0,1:154,2:601,3:834,4:978,5:1292,6:1589,7:1859,8:2659,9:3049,10:3852,11:4146,12:4519,13:4597,14:4937,15:5044,16:5419,17:5628,18:5773,19:6112,20:7171}

    if args.group:
        imagedir =  trackdirid[args.group] # imagedir 第几组
    if args.detectdir:
        argdetectdir=args.detectdir
        assert os.path.exists("./results/%s/%s.txt"%(argdetectdir,imagedir))== True, 'detection dir does not exist!'
        objects3ds = read_det_file("./results/%s/%s.txt"%(argdetectdir,imagedir),type='detection')#sassd 在tracking 数据集探测结果第imagedir组
    else:
        objects3ds=None
    if args.show_tracking:#assign tracking result
        if args.trackdir is None:
            raise Exception("you need to assign --trackdir with your tarcking results!")
        trackdir=args.trackdir
        assert os.path.exists("./results/%s/data/%s.txt"%(trackdir,imagedir))==True,'tracking dir does not exist!'
        track3ds = read_det_file("./results/%s/data/%s.txt"%(trackdir,imagedir),type='tracking')#sassd 在tracking 数据集tracking结果第imagedir组
    else:
        track3ds=None


    if args.show_lidar_with_depth:
        import mayavi.mlab as mlab

        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
        )
    print('all the framid number=%d' % len(dataset))#ly


    for data_idx in range(beginid[int(imagedir)],len(dataset)):  #####测得en(dataset)为8008帧，trainging下所有帧
        timemid=data_idx

        if args.ind > 0:
            data_idx = args.ind  #从设定的那帧开始共8008帧  第几行
        trainvalid=linecache.getline('./data/trainval.txt', data_idx+1)#ly

        data_idx=trainvalid.replace('\n','')  #为了索引相关的点云和图像文件
        frameid = data_idx.split('-')#used for spilt frameid
        # Load data from dataset
        if args.split == "training":
            objects = dataset.get_label_objects(data_idx) #read from relevent label
        else:
            objects = []
        #objects2d = objects2ds[data_idx]
        if int(frameid[1]) not in objects3ds.keys() or int(frameid[0]) != int(imagedir):#不在当前组不用加载结果
            objects3d=None
        else:
            objects3d = objects3ds[int(frameid[1])] # 获得相应帧3ddetection的结果

        if int(frameid[1]) not in track3ds.keys() or int(frameid[0]) != int(imagedir):
            track3d=None
        else:
            track3d = track3ds[int(frameid[1])] # 获得相应帧3dtracking的结果

        objects_pred = None
        track_pred = None
        if args.pred:       #when -p
            # if not dataset.isexist_pred_objects(data_idx):
            #    continue
            #objects_pred = dataset.get_pred_objects(data_idx)
            objects_pred = objects3d # 获得相应帧3ddetection的结果
            if objects_pred == None:
                print('current frame do not have detection results')
        if objects_pred == None:
            print("no pred file")
            # objects_pred[0].print_object()
        if args.show_tracking:  #when --show_tracking
            track_pred=track3d
            if track_pred==None:
                print('current frame do not have tracking results')
        if track_pred == None:
            print("do not load tracking result,if you want to show trackingresults,please use --show_tracking command ")



        n_vec = 4
        if args.pc_label:
            n_vec = 5

        dtype = np.float32
        if args.dtype64:
            dtype = np.float64
        pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]
        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        print(data_idx, "image shape: ", img.shape)
        print(data_idx, "velo  shape: ", pc_velo.shape)
        if args.depth:
            depth, is_exist = dataset.get_depth(data_idx)
            print(data_idx, "depth shape: ", depth.shape)
        else:
            depth = None

        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        # depth_height, depth_width, depth_channel = img.shape

        # print(('Image shape: ', img.shape))

        if args.stat:
            stat_lidar_with_boxes(pc_velo, objects, calib)
            continue
        print("======== Objects in Ground Truth ========")
        n_obj = 0
        for obj in objects:
            if obj.type != "DontCare":
                print("=== {} object ===".format(n_obj + 1))
                obj.print_object()
                n_obj += 1
        print('the current frame is %s' % trainvalid)
        # Draw 3d box in LiDAR point cloud
        if args.show_lidar_topview_with_boxes:
            # Draw lidar top view
            show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred,track_pred)

        # show_image_with_boxes_3type(img, objects, calib, objects2d, data_idx, objects_pred)
        if args.show_image_with_boxes:
            # Draw 2d and 3d boxes on image
            #show_image_with_boxes(img, objects, calib, True, depth)# objects is gt
            show_image_with_boxes(img, objects_pred, calib, show3d=True, depth=depth,track_pred=track_pred) #show3d 是否显示tracking结果
        if args.show_lidar_with_depth:
            # Draw 3d box in LiDAR point cloud
            show_lidar_with_depth(
                pc_velo,
                objects,
                calib,
                fig,
                args.img_fov,
                img_width,
                img_height,
                objects_pred,
                depth,
                img,
                constraint_box=args.const_box,
                save=args.save_depth,
                pc_label=args.pc_label,
            )
            # show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height, \
            #    objects_pred, depth, img)
        if args.show_lidar_on_image:
            # Show LiDAR points on image.
            show_lidar_on_image(pc_velo[:, 0:3], img, calib, img_width, img_height)

        if timemid==beginid[int(imagedir)]:
            input_str = raw_input() #first input enter  if want \enter 假如想每次按键进行则用它

        cv2.waitKey(timeintervel) #ly maybe 50ms show the bb



        #mlab.clf()#图像控制函数mlab，清空当前图像 mlab.clf(figure=None)  感觉没用

        # if input_str == "killall":
        #     break


def depth_to_lidar_format(root_dir, args):
    dataset = kitti_object(root_dir, split=args.split, args=args)
    for data_idx in range(len(dataset)):
        # Load data from dataset

        pc_velo = dataset.get_lidar(data_idx)[:, 0:4]
        calib = dataset.get_calibration(data_idx)
        depth, is_exist = dataset.get_depth(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        print(data_idx, "image shape: ", img.shape)
        print(data_idx, "velo  shape: ", pc_velo.shape)
        print(data_idx, "depth shape: ", depth.shape)
        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        # depth_height, depth_width, depth_channel = img.shape

        # print(('Image shape: ', img.shape))
        save_depth(
            data_idx,
            pc_velo,
            calib,
            args.img_fov,
            img_width,
            img_height,
            depth,
            constraint_box=args.const_box,
        )
        input_str = raw_input()


def read_det_file(det_filename,type='detection'):
    #####""" Parse lines in 2D detection output files """
    """ Parse lines in 3D detection output files """
    det_id2str = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}
    objects = {}
    with open(det_filename, "r") as f:
        if type=='detection':
            for line in f.readlines():
                #obj = utils.Object2d(line.rstrip())
                obj = utils.Object_result_3d(line.rstrip())
                # if obj.img_name not in objects.keys():
                #     objects[obj.img_name] = []
                # objects[obj.img_name].append(obj)
                if obj.image_id not in objects.keys(): #ly
                    objects[obj.image_id] = []
                objects[obj.image_id].append(obj)
            # objects = [utils.Object2d(line.rstrip()) for line in f.readlines()]
        elif type=='tracking':
            for line in f.readlines():
                #obj = utils.Object2d(line.rstrip())
                obj = utils.tracking3d(line.rstrip())
                # if obj.img_name not in objects.keys():
                #     objects[obj.img_name] = []
                # objects[obj.img_name].append(obj)
                if obj.image_id not in objects.keys(): #ly
                    objects[obj.image_id] = []
                objects[obj.image_id].append(obj)
            # objects = [utils.Object2d(line.rstrip()) for line in f.readlines()]
    return objects


if __name__ == "__main__":
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    parser = argparse.ArgumentParser(description="KIITI Object Visualization")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="data/object",
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-i",
        "--ind",
        type=int,
        default=0,
        metavar=0,
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-p", "--pred", action="store_true", help="show predict results"
    )
    parser.add_argument(
        "-s",
        "--stat",
        action="store_true",
        help=" stat the w/h/l of point cloud in gt bbox",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="use training split or testing split (default: training)",
    )
    parser.add_argument(
        "-l",
        "--lidar",
        type=str,
        default="velodyne",
        metavar="N",
        help="velodyne dir  (default: velodyne)",
    )
    parser.add_argument(
        "-e",
        "--depthdir",
        type=str,
        default="depth",
        metavar="N",
        help="depth dir  (default: depth)",
    )
    parser.add_argument(
        "-r",
        "--preddir",
        type=str,
        default="pred",
        metavar="N",
        help="predicted boxes  (default: pred)",
    )
    parser.add_argument("--gen_depth", action="store_true", help="generate depth")
    parser.add_argument("--vis", action="store_true", help="show images")
    parser.add_argument("--depth", action="store_true", help="load depth")
    parser.add_argument("--img_fov", action="store_true", help="front view mapping")
    parser.add_argument("--const_box", action="store_true", help="constraint box")
    parser.add_argument(
        "--save_depth", action="store_true", help="save depth into file"
    )
    parser.add_argument(
        "--pc_label", action="store_true", help="5-verctor lidar, pc with label"
    )
    parser.add_argument(
        "--dtype64", action="store_true", help="for float64 datatype, default float64"
    )

    parser.add_argument(
        "--show_lidar_on_image", action="store_true", help="project lidar on image"
    )
    parser.add_argument(
        "--show_lidar_with_depth",
        action="store_true",
        help="--show_lidar, depth is supported",
    )
    parser.add_argument(
        "--show_image_with_boxes", action="store_true", help="show lidar"
    )
    parser.add_argument(
        "--show_lidar_topview_with_boxes",
        action="store_true",
        help="show lidar topview",
    )
    parser.add_argument(
        "--group",
        type=int,
        default=0000,
        metavar="0000",
        help="input  (input which group,default=0000)",
    )
    parser.add_argument(
        "--detectdir",
        type=str,
        default='saresult_val',
        metavar="saresult_val",
        help="input  (input from which dir from ./results,default='saresult_val')",
    )
    parser.add_argument(
        "--show_tracking",action="store_true", help="show tracking results,now only show on lidarBEV"
    )
    parser.add_argument(
        "--trackdir",
        type=str,
        default=None,
        metavar="carsassd_tra_val",
        help="input  (input tracking results (follow ab3d output format) from which dir from ./results/,default=None)",
    )

    args = parser.parse_args()
    if args.pred:
        assert os.path.exists(args.dir + "/" + args.split + "/pred")

    if args.vis:
        dataset_viz(args.dir, args)
    if args.gen_depth:
        depth_to_lidar_format(args.dir, args)
