from diffusers import StableDiffusionImg2ImgPipeline
from get_input import get_img
import torch

model_path = "pretrained"
#读取模型
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path,torch_dtype=torch.float32).to("cuda")
#图片url
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#提示词prompt设置
text="((two)) ((dogs)) in the picture, (((nature)), (((beauty))), (((smooth)))，white，Highest quality"
#获取图像
raw_image=get_img(url=url,local_img=False)
#模型正向传播
image = pipe(image=raw_image,prompt=text,strength=0.8, guidance_scale=7.5).images[0]
#图片保存
image.save("reproducibility/sd2.jpg")