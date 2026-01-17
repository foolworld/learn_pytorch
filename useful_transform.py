from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")

img = Image.open("dataset/val/ants/8398478_50ef10c47a.jpg")
print(img)

#totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("totensor",img_tensor)

#normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)
writer.close()
