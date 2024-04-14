
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
import threading
import concurrent.futures
import mmengine
import yaml
import time
from easydict import EasyDict

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        model_root = "./output/dynerf"
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        dataset.model_path = os.path.join(model_root, "cook_spinach")
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        views = scene.getVideoCameras()

        save_path = pipeline.expname
        name = "video"
        render_path = os.path.join(save_path, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(save_path, "ours_{}".format(iteration), "gt")
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        render_images = []
        gt_list = []
        render_list = []

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            if idx == 0:
                del gaussians
                del scene
                time1 = time.time()
                print("Loading 2s01")
                dataset.model_path = os.path.join(model_root, "cook_spinach_2s01")
                gaussians = GaussianModel(dataset.sh_degree, hyperparam)
                print("point nums:",gaussians._xyz.shape[0])
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
                cam_type=scene.dataset_type
            elif idx == 60:
                # free memory
                del gaussians
                del scene
                print("Loading 2s02")
                dataset.model_path = os.path.join(model_root, "cook_spinach_2s02")
                gaussians = GaussianModel(dataset.sh_degree, hyperparam)
                print("point nums:",gaussians._xyz.shape[0])
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
                cam_type=scene.dataset_type
            elif idx == 120:
                del gaussians
                del scene
                print("Loading 2s03")
                dataset.model_path = os.path.join(model_root, "cook_spinach_2s03")
                gaussians = GaussianModel(dataset.sh_degree, hyperparam)
                print("point nums:",gaussians._xyz.shape[0])
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
                cam_type=scene.dataset_type
            elif idx == 180:
                del gaussians
                del scene
                print("Loading 2s04")
                dataset.model_path = os.path.join(model_root, "cook_spinach_2s04")
                gaussians = GaussianModel(dataset.sh_degree, hyperparam)
                print("point nums:",gaussians._xyz.shape[0])
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
                cam_type=scene.dataset_type
            elif idx == 240:
                del gaussians
                del scene
                print("Loading 2s05")
                dataset.model_path = os.path.join(model_root, "cook_spinach_2s05")
                gaussians = GaussianModel(dataset.sh_degree, hyperparam)
                print("point nums:",gaussians._xyz.shape[0])
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
                cam_type=scene.dataset_type
            else:
                pass

            rendering = render(view, gaussians, pipeline, background, cam_type=cam_type)["render"]
            render_images.append(to8b(rendering).transpose(1,2,0))
            render_list.append(rendering)

        time2=time.time()
        print("FPS:",(len(views)-1)/(time2-time1))

        multithread_write(gt_list, gts_path)
        multithread_write(render_list, render_path)

        
        # Use imageio to write render_images to a video.mp4
        video_writer = imageio.get_writer(os.path.join(save_path, "ours_{}".format(iteration), 'video_rgb.mp4'), fps=30)
        for image in render_images:
            video_writer.append_data(image)
        video_writer.close()



if __name__ == "__main__":

    parser = ArgumentParser(description="Rendering script with YAML config")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    args = EasyDict(config)
    
    render_sets(args.model_params, args.model_hidden_params, args.iteration, args.pipeline_params)
