
from donut import DonutModel
from PIL import Image
import torch
model = DonutModel.from_pretrained("/content/donut/result/train_ben/20220917_221154")
if torch.cuda.is_available():
    model.half()
    device = torch.device("cuda")
    model.to(device)
else:
    model.encoder.to(torch.bfloat16)
model.eval()
image = Image.open("dataset/ben-donut/train/001.jpg").convert("RGB")
output = model.inference(image=image, prompt="<s_sroie-donut>")

print(output)