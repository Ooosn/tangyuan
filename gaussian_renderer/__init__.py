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
import math
from v_3dgs import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import numpy as np
from utils.graphics_utils import getProjectionMatrixWithPrincipalPoint, look_at, getProjectionMatrix
from diff_gaussian_rasterization_light import GaussianRasterizationSettings as gs_light_settings
from diff_gaussian_rasterization_light import  GaussianRasterizer as gs_light_rasterizer

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
           separate_sh = False, override_color = None, use_trained_exp=False,
           updated_light_pos=None, offset = 0.6):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)



    # light

    # calculate the fov and projmatrix of light
    fx_origin = viewpoint_camera.image_width / (2. * tanfovx)
    fy_origin = viewpoint_camera.image_height / (2. * tanfovy)

    # calculate the fov for shadow splatting:
    # 计算光源和相机的距离比 f_scale_ratio
    print(pc.get_xyz.mean(dim=0))
    max_xyz = pc.get_xyz.max(dim=0).values  # shape: (3,)
    min_xyz = pc.get_xyz.min(dim=0).values  # shape: (3,)
    range_xyz = max_xyz - min_xyz
    print("extent", range_xyz)
    if updated_light_pos is None:
        light_position = pc.get_xyz.mean(dim=0).detach().cpu().numpy() + (25, 25, 50)
    else:
        light_position = pc.get_xyz.mean(dim=0).detach().cpu().numpy() + updated_light_pos
    print(light_position)
    camera_position = viewpoint_camera.camera_center.detach().cpu().numpy()
    f_scale_ratio = np.sqrt(np.sum(light_position * light_position) / np.sum(camera_position * camera_position))
    print("f_scale_ratio", f_scale_ratio)
    # 计算光源的焦距
    fx_far = fx_origin * f_scale_ratio
    fy_far = fy_origin * f_scale_ratio
    cx = viewpoint_camera.image_width / 2.0
    cy = viewpoint_camera.image_height / 2.0

    # 先计算 FoV 对应的正切值，然后通过 arctan 得出 FoV
    tanfovx_far = 0.5 * viewpoint_camera.image_width / fx_far
    tanfovy_far = 0.5 * viewpoint_camera.image_height / fy_far
    # 将焦距反推回视场角（FoV）
    fovx_far = 2 * math.atan(tanfovx_far)
    fovy_far = 2 * math.atan(tanfovy_far)
    object_center=pc.get_xyz.mean(dim=0).detach()
    world_view_transform_light=look_at(light_position,
                                       object_center.detach().cpu().numpy(),
                                       up_dir=np.array([0, 0, 1]))
    world_view_transform_light=torch.tensor(world_view_transform_light,
                                            device=viewpoint_camera.world_view_transform.device,
                                            dtype=viewpoint_camera.world_view_transform.dtype)   
    
    light_prjection_matrix = getProjectionMatrix(znear=viewpoint_camera.znear, zfar=viewpoint_camera.zfar, fovX=fovx_far, fovY=fovy_far).transpose(0,1).cuda()
    full_proj_transform_light = (world_view_transform_light.unsqueeze(0).bmm(light_prjection_matrix.unsqueeze(0))).squeeze(0)
    
    # 设置光源的高斯泼溅参数
    raster_settings_light = gs_light_settings(
        image_height = 2*int(viewpoint_camera.image_height),
        image_width = 2*int(viewpoint_camera.image_width),
        tanfovx = tanfovx_far,
        tanfovy = tanfovy_far,
        bg = bg_color[:3],
        scale_modifier = scaling_modifier,
        viewmatrix = world_view_transform_light,
        projmatrix = full_proj_transform_light,
        sh_degree = pc.active_sh_degree,
        campos = torch.tensor(light_position, dtype=torch.float32, device='cuda'),
        prefiltered = False,
        debug = pipe.debug,
        low_pass_filter_radius = 0.3,
    )

    rasterizer_light = gs_light_rasterizer(raster_settings=raster_settings_light)
            
    opacity_light = torch.zeros([pc.get_xyz.shape[0],1], dtype=torch.float32, device='cuda')
    means3D = pc.get_xyz
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3Ds_precomp = None
    if pipe.compute_cov3D_python:
        cov3Ds_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    

    light_inputs = {
                # 高斯点相关
                "means3D": means3D,
                "means2D": screenspace_points,
                "shs": None,
                "colors_precomp": torch.zeros((2, 3), dtype=torch.float32, device='cuda'),
                "opacities": opacity,
                "scales": scales,
                "rotations": rotations,
                "cov3Ds_precomp": cov3Ds_precomp,

                # 阴影相关
                "non_trans": opacity_light,
                "offset": offset,
                "thres": -1,

                # prune 相关
                "is_train": False,
                
                # hgs 相关
                "hgs": False,
                "hgs_normals": None,
                "hgs_opacities": None,
                "hgs_opacities_shadow": None, 
                "hgs_opacities_light": None, 

                # 流
                "streams": None # 暂时没用，（用于内部多个流）

            }
    _, out_weight, _, shadow = rasterizer_light(**light_inputs)
    opacity_light1 = torch.clamp_min(opacity_light, 1e-6)
    shadow = shadow / opacity_light1 
    print((opacity_light<1e-4).sum())
    shadow[opacity_light<1e-4] = 1
    assert not torch.isnan(shadow).any()


    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    cov3D_precomp = None

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color
    print(shadow.shape)
    print(pc.get_xyz.shape)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            shadow = shadow
            )
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            shadow = shadow
            )
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image[0:3, :, :],
        "shadow": rendered_image[3, :, :],
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
