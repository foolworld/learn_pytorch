from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#绝对路径:C:\Users\xs\Desktop\learn_pytorch\dataset\train\ants_image\7759525_1363d24e88.jpg
#相对路径dataset/train/ants_image/7759525_1363d24e88.jpg
img_path="dataset/train/ants_image/7759525_1363d24e88.jpg"
img =Image.open(img_path)

writer = SummaryWriter("logs")

tensor = transforms.ToTensor()
tensor_img = tensor(img)

writer.add_image("Tensor_img", tensor_img)
print(img.size )

writer.close()