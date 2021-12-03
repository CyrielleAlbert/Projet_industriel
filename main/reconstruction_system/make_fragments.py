# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/make_fragments.py

import math
import sys
import numpy as np
import open3d as o3d
sys.path.append("../utility")
from file import join, make_clean_folder, get_rgbd_file_lists
from opencv import initialize_opencv
sys.path.append(".")
from optimize_posegraph import optimize_posegraph_for_fragment
from os import listdir
# check opencv python package
with_opencv = initialize_opencv()
if with_opencv:
    from opencv_pose_estimation import pose_estimation
import cv2
from PIL import Image

def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity, config):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        convert_rgb_to_intensity=convert_rgb_to_intensity)
    #visualisation RGBD images
    #intrinsic = o3d.camera.PinholeCameraIntrinsic(width=960,height=1280,cx=480,cy=640,fx=500,fy=500)
    #point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,intrinsic)
    #o3d.visualization.draw_geometries([point_cloud])
    return rgbd_image


def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                           with_opencv, config):
            
    source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], True,
                                        config)
    target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], True,
                                        config)

    option = o3d.pipelines.odometry.OdometryOption()
    option.max_depth_diff = config["max_depth_diff"]
    if abs(s - t) != 1:
        if with_opencv:
            success_5pt, odo_init = pose_estimation(source_rgbd_image,
                                                    target_rgbd_image,
                                                    intrinsic, False)
            if success_5pt:
                [success, trans, info
                ] = o3d.pipelines.odometry.compute_rgbd_odometry(
                    source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option)
                return [success, trans, info]
        return [False, np.identity(4), np.identity(6)]
    else:
        odo_init = np.identity(4)
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        return [success, trans, info]


def make_posegraph_for_fragment(path_dataset, sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(trans_odometry))
    for s in range(sid, eid):
        for t in range(s + 1, eid):
            # odometry
            if t == s + 1:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        trans_odometry_inv))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(s - sid,
                                                             t - sid,
                                                             trans,
                                                             info,
                                                             uncertain=False))

            # keyframe loop closure
            if s % config['n_keyframes_per_n_frame'] == 0 \
                    and t % config['n_keyframes_per_n_frame'] == 0:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            s - sid, t - sid, trans, info, uncertain=True))
    o3d.io.write_pose_graph(
        join(path_dataset, config["template_fragment_posegraph"] % fragment_id),
        pose_graph)


def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id,
                                      n_fragments, pose_graph_name, intrinsic,
                                      config):
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"]/512,
        #voxel_length=1,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    rgbds = []
    for i in range(len(pose_graph.nodes)):
        i_abs = fragment_id * config['n_frames_per_fragment'] + i
        print(
            "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
            (fragment_id, n_fragments - 1, i_abs, i + 1, len(pose_graph.nodes)))

        color_arr = o3d.io.read_image(color_files[i_abs])
        depth_arr = o3d.io.read_image(depth_files[i_abs])
        print(np.max(depth_arr))
        rgbd =  o3d.geometry.RGBDImage.create_from_color_and_depth(color_arr,depth_arr,depth_scale=1,depth_trunc= np.inf,convert_rgb_to_intensity=False)
        print(np.max(np.asarray(rgbd.depth)))
        pose = pose_graph.nodes[i].pose
        extrinsic = np.linalg.inv(pose)
        print("\nextrinsic: ",extrinsic,"\n\n")
        print("intrinsic: ",intrinsic,"\n\n")
        print("RGBDImage: ",rgbd,"\n")
        volume.integrate(rgbd, intrinsic, extrinsic)
        point_cloud =  volume.extract_point_cloud()
        #o3d.visualization.draw_geometries([point_cloud])
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    return mesh


def make_pointcloud_for_fragment(path_dataset, color_files, depth_files,
                                 fragment_id, n_fragments, intrinsic, config):
    mesh = integrate_rgb_frames_for_fragment(
        color_files, depth_files, fragment_id, n_fragments,
        join(path_dataset,
             config["template_fragment_posegraph_optimized"] % fragment_id),
        intrinsic, config)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd_name = join(path_dataset,
                    config["template_fragment_pointcloud"] % fragment_id)
    hi = o3d.io.write_point_cloud(pcd_name, pcd)


def process_single_fragment(fragment_id, color_files, depth_files, n_files,
                            n_fragments, config):
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=config["intrinsic_info"]["width"],
                                                      height=config["intrinsic_info"]["height"],
                                                      cx=config["intrinsic_info"]["cx"],
                                                      cy=config["intrinsic_info"]["cy"],
                                                      fx=config["intrinsic_info"]["f"],
                                                      fy=config["intrinsic_info"]["f"])
    sid = fragment_id * config['n_frames_per_fragment']
    eid = min(sid + config['n_frames_per_fragment'], n_files)

    make_posegraph_for_fragment(config["path_dataset"], sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config)
    optimize_posegraph_for_fragment(config["path_dataset"], fragment_id, config)
    make_pointcloud_for_fragment(config["path_dataset"], color_files,
                                 depth_files, fragment_id, n_fragments,
                                 intrinsic, config)


def run(config):

    print("making fragments from RGBD sequence.")
    make_clean_folder(join(config["path_dataset"], config["folder_fragment"]))
    images_names = listdir(config["path_dataset"]+"/image/")
    try:
        images_names.remove("Thumbs.db")
    except:
        print("no Thunbs.db")
    color_files = np.array([config["path_dataset"]+'image/'+ img_name for img_name in images_names])[0:99].tolist()
    depth_files = np.array([config["path_dataset"]+'depth2/'+ img_name for img_name in images_names])[0:99].tolist()   

    n_files = len(color_files)
    n_fragments = int(
        math.ceil(float(n_files) / config['n_frames_per_fragment']))

    if config["python_multi_threading"] is True:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(), n_fragments)
        Parallel(n_jobs=MAX_THREAD)(delayed(process_single_fragment)(
            fragment_id, color_files, depth_files, n_files, n_fragments, config)
                                    for fragment_id in range(n_fragments))
    else:
        for fragment_id in range(n_fragments):
            process_single_fragment(fragment_id, color_files, depth_files,
                                    n_files, n_fragments, config)
