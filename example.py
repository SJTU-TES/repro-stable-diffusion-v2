from diffusers import StableDiffusionImg2ImgPipeline
from get_input import get_img
import torch

model_path = "./img2img"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path,torch_dtype=torch.float16,revision="fp16",use_auth_token=True)
pipe=pipe.to("cuda")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
text="Generate a picture in which three dogs run"
raw_image=get_img(url=url,local_img=False)
image = pipe(image=raw_image,prompt=text,strength=0.8, guidance_scale=7.5).images[0]