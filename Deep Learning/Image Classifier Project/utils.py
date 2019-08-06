import json

from PIL import Image
import numpy as np
from torchvision import transforms


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225]
                                                         )
                                    ])

    image = transform(pil_image)
    np_image = np.array(image)

    return np_image


def load_cat_to_name(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
