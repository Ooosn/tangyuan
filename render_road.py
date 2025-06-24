#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal
from sklearn.decomposition import PCA
import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import *
from flask import Flask, send_file

# Define a global light position (initial value same as your script)
app = Flask(__name__)
@app.route('/rendered')
def get_rendered_image():
    return send_file('/tmp/output.png', mimetype='image/png')


def init_window_and_imgui(width=960, height=540, title="Light Control"):
    if not glfw.init():
        raise Exception("GLFW initialization failed")

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")

    glfw.make_context_current(window)
    imgui.create_context()
    impl = GlfwRenderer(window)

    return window, impl
    
def upload_image_to_texture(image_np, texture_id):
    h, w, _ = image_np.shape
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, image_np)


def compute_centroid_and_orientation(point_cloud):
    """
    ?????????????RT???

    ??:
        point_cloud (np.ndarray): ???????? (N, 3)

    ??:
        centroid (np.ndarray): ???????? (3,)
        RT (np.ndarray): ?????-???????? (4, 4)
    """
    # ????
    centroid = np.mean(point_cloud, axis=0)
    
    # ??PCA??????
    pca = PCA(n_components=3)
    pca.fit(point_cloud - centroid)  # ?????
    
    # PCA??????
    rotation = pca.components_.T  # ????????
    
    # ???????????1?????????
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] = -rotation[:, 2]
    
    # ??RT??
    # RT_colmap = np.eye(4)
    # RT_colmap[:3, :3] = rotation.T
    # RT_colmap[:3, 3] = -rotation.T @ centroid

    RT = np.eye(4)
    RT[:3, :3] = rotation
    RT[:3, 3] = centroid
    
    return torch.tensor(centroid), torch.tensor(RT)

def render_sets_gui(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(2)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        tamplete_cam = scene.getTrainCameras()[0]
        bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        camera_trajectorys = np.load(os.path.join(dataset.model_path, 'single_view/camera.npy'),  allow_pickle=True)

        render_path = os.path.join(dataset.model_path, 'single_view', "clean_scene_v2")
        os.makedirs(render_path,exist_ok=True)

        gaussians_combine = GaussianModel(2)
        left_car_path = os.path.join(dataset.model_path, 'single_view', 'transform_vehicle_gaussians_left.ply')
        right_car_path = os.path.join(dataset.model_path, 'single_view', 'transform_vehicle_gaussians_right.ply')
        scene_path = os.path.join(dataset.model_path, 'single_view', 'clean_scene_v2.ply')
        gaussians_combine.load_ply_combine(left_car_path, right_car_path, scene_path)

        idx = 0

    
        window, impl = init_window_and_imgui()

        light_position_delta = [25.0, 25.0, 50.0]
        texture_id = glGenTextures(1)
        app = Flask(__name__)


        for camera_trajectory in camera_trajectorys: pass
        camera_trajectory = camera_trajectorys[0]
        while not glfw.window_should_close(window):
            glfw.poll_events()
            impl.process_inputs()

            imgui.new_frame()

            # --- 控制面板 ---
            imgui.begin("Light Control")
            changed_x, light_position_delta[0] = imgui.slider_float("Light X", light_position_delta[0], -100, 100)
            changed_y, light_position_delta[1] = imgui.slider_float("Light Y", light_position_delta[1], -100, 100)
            changed_z, light_position_delta[2] = imgui.slider_float("Light Z", light_position_delta[2], -100, 100)

            if imgui.button("Render Next View"):
                do_render = True
            else:
                do_render = False

            imgui.end()
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            impl.render(imgui.get_draw_data())       

            if do_render:
                extrinsic_matrix = camera_trajectory['extrinsic_matrix']
                intrinsic_matrix = camera_trajectory['intrinsic_matrix']
                fx, fy = intrinsic_matrix[0,0] * 0.5 , intrinsic_matrix[1,1] * 0.5
                cx, cy = intrinsic_matrix[0,2], intrinsic_matrix[1,2]
                FovY = focal2fov(fy, int(cy*2))
                FovX = focal2fov(fx, int(cx*2))
                R = extrinsic_matrix[:3,:3]
                T = extrinsic_matrix[:3,3]
                # import pdb;pdb.set_trace()

                from torchvision.transforms import ToPILImage
                image_tensor = tamplete_cam.original_image.cpu()  # 确保在 CPU 上
                image = ToPILImage()(image_tensor)                # 转为 PIL.Image
                print("1111111")
                cam = Camera(
                    resolution = (960, 540),
                    colmap_id=tamplete_cam.colmap_id, 
                    R=R, T=T, 
                    FoVx=FovX, FoVy=FovY, 
                    depth_params=None,
                    image=image,
                    invdepthmap = None,
                    image_name=f"view_{idx}.jpg", 
                    uid=tamplete_cam.uid, 
                    data_device=args.data_device)
                
                vehicle_render_pkg = render(cam, gaussians_combine, pipeline, background, updated_light_pos=light_position_delta)
                render_result = vehicle_render_pkg["render"].cpu()
                shadow_result = vehicle_render_pkg["shadow"].cpu()
                # torchvision.utils.save_image(shadow_result, os.path.join(render_path, '{0:05d}'.format(int(idx)) + "_shadow.png"))
                # torchvision.utils.save_image(render_result, os.path.join(render_path, '{0:05d}'.format(int(idx)) + ".png"))
                idx = 1
                print("You triggered one frame render with light:", light_position_delta)
                if idx == 10:
                    break
                image_np = (shadow_result.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # shape = (H, W, 3)  

                app.run(host='0.0.0.0', port=8000)
                upload_image_to_texture(image_np, texture_id)

            imgui.begin("Rendered Image")
            imgui.image(texture_id, image_np.shape[1], image_np.shape[0])
            imgui.end()


            glfw.swap_buffers(window)           
        # 退出清理
        impl.shutdown()
        glfw.terminate()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(2)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        tamplete_cam = scene.getTrainCameras()[0]
        bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        camera_trajectorys = np.load(os.path.join(dataset.model_path, 'single_view/camera.npy'),  allow_pickle=True)

        render_path = os.path.join(dataset.model_path, 'single_view', "clean_scene_v2")
        os.makedirs(render_path,exist_ok=True)

        gaussians_combine = GaussianModel(2)
        left_car_path = os.path.join(dataset.model_path, 'single_view', 'transform_vehicle_gaussians_left.ply')
        right_car_path = os.path.join(dataset.model_path, 'single_view', 'transform_vehicle_gaussians_right.ply')
        scene_path = os.path.join(dataset.model_path, 'single_view', 'clean_scene_v2.ply')
        gaussians_combine.load_ply_combine(left_car_path, right_car_path, scene_path)

        idx = 0

        light_position_deltas = ([25.0, 25.0, 500.0], [50.0, 50.0, 1000.0], [-50.0, -50.0, 1000.0],
                                [-25.0, -25.0, 1000.0], [-50.0, 50.0, 1500.0], [50.0, -50.0, 1500.0],
                                [0, 0, 500], [0, 0 ,1000])
        


        for camera_trajectory in camera_trajectorys: pass
        camera_trajectory = camera_trajectorys[0]

        for x, y, z in light_position_deltas:
            extrinsic_matrix = camera_trajectory['extrinsic_matrix']
            intrinsic_matrix = camera_trajectory['intrinsic_matrix']
            fx, fy = intrinsic_matrix[0,0] * 0.5 , intrinsic_matrix[1,1] * 0.5
            cx, cy = intrinsic_matrix[0,2], intrinsic_matrix[1,2]
            FovY = focal2fov(fy, int(cy*2))
            FovX = focal2fov(fx, int(cx*2))
            R = extrinsic_matrix[:3,:3]
            T = extrinsic_matrix[:3,3]
            # import pdb;pdb.set_trace()

            from torchvision.transforms import ToPILImage
            image_tensor = tamplete_cam.original_image.cpu()  # 确保在 CPU 上
            image = ToPILImage()(image_tensor)                # 转为 PIL.Image
            print("1111111")
            cam = Camera(
                resolution = (960, 540),
                colmap_id=tamplete_cam.colmap_id, 
                R=R, T=T, 
                FoVx=FovX, FoVy=FovY, 
                depth_params=None,
                image=image,
                invdepthmap = None,
                image_name=f"view_{idx}.jpg", 
                uid=tamplete_cam.uid, 
                data_device=args.data_device)
            light_position_delta = [x, y, z]
            vehicle_render_pkg = render(cam, gaussians_combine, pipeline, background, 
                                        updated_light_pos=light_position_delta, offset=0.3)
            render_result = vehicle_render_pkg["render"].cpu()
            shadow_result = vehicle_render_pkg["shadow"].cpu()
            torchvision.utils.save_image(shadow_result, os.path.join(render_path, '{0:05d}_{1}_{2}_{3}'.format(int(idx), x, y, z) + "_shadow.png"))
            torchvision.utils.save_image(render_result, os.path.join(render_path, '{0:05d}_{1}_{2}_{3}'.format(int(idx), x, y, z) + ".png"))
            idx = 1
            if idx == 10:
                pass  




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)