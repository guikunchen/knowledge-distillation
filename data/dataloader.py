from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import DatasetFolder
from utils.pseudo_labels import get_pseudo_labels
from torchvision import transforms


def _get_tfm(img_size):
    train_tfm = transforms.Compose([
        # Resize the image into a fixed shape (height = width = img_size)
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomCrop(128),
        transforms.ToTensor(),
    ])

    # We don't need augmentations in testing and validation.
    # All we need here is to resize the PIL image and transform it into Tensor.
    test_tfm = transforms.Compose([
        # Resize the image into a fixed shape (height = width = img_size)
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
    ])

    return train_tfm, test_tfm


def _get_dataset():
    train_tfm, test_tfm = _get_tfm(142)

    # Construct datasets.
    # The argument "loader" tells how torchvision reads the data.
    train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg",
                              transform=train_tfm)
    valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg",
                              transform=test_tfm)
    unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg",
                                  transform=train_tfm)
    test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    return train_set, valid_set, unlabeled_set, test_set


def get_dataloader(batch_size=64, do_semi=False, teacher_net=None):
    train_set, valid_set, unlabeled_set, test_set = _get_dataset()
    # Construct data loaders.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if do_semi:
        # Generate new trainloader with unlabeled set.
        unlabeled_set = get_pseudo_labels(unlabeled_set, teacher_net, batch_size)
        concat_dataset = ConcatDataset([train_set, unlabeled_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    return train_loader, valid_loader, test_loader
