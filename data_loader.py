from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

batch_size = 64

# MNIST
mnist_train = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
mnist_loader_train = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
mnist_loader_test = DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# MNIST-M
# TODO

# MNIST-I
mnist_inverted_train = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
mnist_inverted_test = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
mnist_inverted_train.data = 255 - mnist_inverted_train.data
mnist_inverted_test.data = 255 - mnist_inverted_test.data
mnist_inverted_loader_train = DataLoader(dataset=mnist_inverted_train, batch_size=batch_size, shuffle=True, drop_last=True)
mnist_inverted_loader_test = DataLoader(dataset=mnist_inverted_test, batch_size=batch_size, shuffle=True, drop_last=True)

# SVHN
svhn_train = datasets.SVHN(root='./data/', split='train', transform=transform, download=True)
svhn_test = datasets.SVHN(root='./data/', split='test', transform=transform, download=True)
svhn_loader_train = DataLoader(dataset=svhn_train, batch_size=batch_size, shuffle=True, drop_last=True)
svhn_loader_test = DataLoader(dataset=svhn_test, batch_size=batch_size, shuffle=True, drop_last=True)

# USPS
usps_train = datasets.USPS(root='./data', train=True, transform=transform, download=True)
usps_test = datasets.USPS(root='./data', train=False, transform=transform, download=True)
usps_loader_train = DataLoader(dataset=usps_train, batch_size=batch_size, shuffle=True, drop_last=True)
usps_loader_test = DataLoader(dataset=usps_test, batch_size=batch_size, shuffle=True, drop_last=True)