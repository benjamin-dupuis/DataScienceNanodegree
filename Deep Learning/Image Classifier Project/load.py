import torch
from torchvision import datasets, transforms


def load_data(data_dir):
    """
    Load the data into training, test and validation dataloaders.
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {"train": transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225]
                                                                         )
                                                    ]),

                       "test": transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225]
                                                                        )
                                                   ]),

                       "valid": transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225]
                                                                         )
                                                    ])
                       }

    image_datasets = {"train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
                      "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
                      "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"])
                      }

    dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=50, shuffle=True),
                   "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=50),
                   "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=50)
                   }

    train_data = dataloaders["train"]
    test_data = dataloaders["test"]
    valid_data = dataloaders["valid"]
    class_to_idx = image_datasets['train'].class_to_idx

    return train_data, test_data, valid_data, class_to_idx
