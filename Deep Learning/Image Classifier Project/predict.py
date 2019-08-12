import argparse
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--category_names', type=str, default=None, required=False)
parser.add_argument('--top_k', type=int, default=1, required=False)
parser.add_argument('--gpu', action="store_true", default=False, required=False)


class Config:
    def __init__(self, args):
        self.image_path = args.image_path
        self.checkpoint = args.checkpoint
        self.top_k = args.top_k
        self.use_gpu = args.gpu
        self.category_names = args.category_names

        cuda_is_available = torch.cuda.is_available()
        if not cuda_is_available:
            print("Cuda is not available. Only the CPU will be used")
        self.device = torch.device("cuda:0" if cuda_is_available and self.use_gpu else "cpu")


def predict(config):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model, mapping = load_trained_model(config.checkpoint)
    model.to(config.device)
    image = torch.from_numpy(process_image(config.image_path)).unsqueeze_(0)
    image = image.to(config.device)
    model.eval()
    with torch.no_grad():
        logps = model.forward(image)
        probabilities = torch.exp(logps)
        top_probabilities, top_indices = probabilities.topk(config.top_k, dim=1)

    top_probabilities = top_probabilities.cpu().numpy().flatten().tolist()
    top_indices = top_indices.cpu().numpy().flatten().tolist()

    # Convert the indices to the actual classes
    top_classes = [k for k, v in mapping.items() if v in top_indices]

    if config.category_names is not None:
        cat_to_name = load_cat_to_name(config.category_names)
        top_classes = [cat_to_name[cat] for cat in top_classes]

    return top_probabilities, top_classes

if __name__ == "__main__":
    arguments = parser.parse_args()
    print(arguments)
    configuration = Config(arguments)
    probs, classes = predict(configuration)
    classes_and_probs = dict(zip(classes, probs))
    print("Predicted classes with their probabilities : ")
    print(classes_and_probs)
