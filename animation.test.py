import os
import numpy as np
import IPython.display
from PIL import Image
from stable_diffusion_pytorch import pipeline, model_loader

# 预加载模型
models = model_loader.preload_models('cpu')

# 生成参数
device = 'cpu'
strength_first = 0.8  # 生成第一帧的强度（较高）
strength_last = 0.2 
do_cfg = True
cfg_scale = 7.5
height = 512
width = 512
sampler = "k_lms"
n_inference_steps = 15
num_frames = 10 

use_seed = False
seed = 42 if use_seed else None

# 创建存储目录
output_folder = 'animation_frames'
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# **1️⃣ 生成第一张图**
first_prompt = "Black and white illustration style,many woolly sheep hear the call and look up to see what's going on,subject sheep zoom in,"
first_image = pipeline.generate(
    prompts=[first_prompt], uncond_prompts=None,
    input_images=[], strength=strength_first,
    do_cfg=do_cfg, cfg_scale=cfg_scale,
    height=height, width=width, sampler=sampler,
    n_inference_steps=n_inference_steps, seed=seed,
    models=models, device=device, idle_device='cpu'
)[0]

# 保存第一帧
first_image.save(f"{output_folder}/frame_00.jpg")

# **2️⃣ 生成最后一张图**
last_prompt = "Black and white illustration style,many woolly sheep eat grass with their head down"
last_image = pipeline.generate(
    prompts=[last_prompt], uncond_prompts=None,
    input_images=[], strength=strength_last,
    do_cfg=do_cfg, cfg_scale=cfg_scale,
    height=height, width=width, sampler=sampler,
    n_inference_steps=n_inference_steps, seed=seed,
    models=models, device=device, idle_device='cpu'
)[0]

# 保存最后一帧
last_image.save(f"{output_folder}/frame_{num_frames}.jpg")

# **3️⃣ 生成中间过渡帧**
for i in range(1, num_frames):
    alpha = i / num_frames  # 计算混合比例（从 0% 到 100%）
    blended_image = Image.blend(first_image, last_image, alpha)  # 图像插值
    num_str = str(i).zfill(2)
    blended_image.save(f"{output_folder}/frame_{num_str}.jpg")

# **4️⃣ 生成 GIF**
os.system("ffmpeg -y -f image2 -framerate 4 -i 'animation_frames/frame_%02d.jpg' -loop 0 ai_animation.gif")

# **5️⃣ 显示 GIF**
IPython.display.Image('ai_animation.gif')
