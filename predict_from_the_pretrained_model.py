

from donut import DonutModel
from PIL import Image
import torch
model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
if torch.cuda.is_available():
    model.half() 
    device = torch.device("cuda") 
    model.to(device) 
else: 
    model.encoder.to(torch.bfloat16)
model.eval() 
image = Image.open("dataset/ben-receipt/validation/1000-receipt.jpg").convert("RGB")
output = model.inference(image=image, prompt="<s_cord-v2>")
print(output)