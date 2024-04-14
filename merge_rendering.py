import imageio
import numpy as np
import os
from tqdm import tqdm
import glob

if __name__ == "__main__":
    rendering_root = "/home/ubuntu/4DGaussians/output/dynerf"
    merge_save_dir = "/home/ubuntu/4DGaussians/output/dynerf/merged"
    frames_save_dir = "/home/ubuntu/4DGaussians/output/dynerf/merged/renders"
    os.makedirs(merge_save_dir, exist_ok=True)
    os.makedirs(frames_save_dir, exist_ok=True)

    all_frame_list = []
    frame_idx = 0
    for i in range(1, 6):
        _dir = os.path.join(rendering_root, "cook_spinach_2s0{}".format(i))
        frame_dir = os.path.join(_dir, "video", "ours_14000", "renders")
        frames = glob.glob(os.path.join(frame_dir, "*.png"))
        frames.sort()
        for fname in frames:
            frame = imageio.imread(fname)
            all_frame_list.append(frame)
      
            fname_newframe = os.path.join(frames_save_dir, '{0:04d}'.format(frame_idx) + ".png")
            imageio.imwrite(fname_newframe, frame)
            frame_idx += 1
    video_writer = imageio.get_writer(os.path.join(merge_save_dir, 'video_rgb.mp4'), fps=30)
    for image in all_frame_list:
        video_writer.append_data(image)
    video_writer.close()
    print("Video saved to {}".format(os.path.join(merge_save_dir, 'video_rgb.mp4')))
        
        
        