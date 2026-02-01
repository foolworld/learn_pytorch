from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvisions import transforms

writer = SummaryWriter("logs")

img = Image.open("dataset/227a5eccc6e58130bb27e231e0b8c784.jpg")
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

#resize
print(img.size)
trans_resize = transforms.Resize((224, 224))
#img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
#img_resize PIL ->  totensor -> img_resize tensor
img_resize =trans_totensor(img_resize)
writer.add_image("Resize", img_resize,0)
print(img_resize)


#compose - resize -2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_totensor, trans_resize_2])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2,1)

#randomcrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("Crop", img_crop,i)

writer.close()