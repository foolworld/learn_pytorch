import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=dataset_transforms)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=dataset_transforms)

# print(test_set[0])
# print(test_set.classes)
#
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
# print(test_set[0])

writer = SummaryWriter("p10")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("Test Image",img,i)
writer.close()