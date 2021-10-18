from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def load_dataset(args):

    # Set Transformations
    transform_train = Compose([
        Resize(C.get()['target_size'] + 32, interpolation=PIL.Image.BICUBIC),
    ])

    # Load dataset
    if args.dataset == 'MNIST':
        trainset = datasets.MNIST(root=imagenet_path, train=training, download=True, transform=transform_train)
    else:
        raise ValueError(dataset)
    return trainset
