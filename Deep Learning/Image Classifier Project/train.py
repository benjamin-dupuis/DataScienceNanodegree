import argparse
from torch import optim
from load import load_data
from model import *
from workspace_utils import active_session

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--arch', type=str, required=True)
parser.add_argument('--learning_rate', type=float, default=0.001, required=False)
parser.add_argument('--hidden_units', type=int, default=512, required=False)
parser.add_argument('--epochs', type=int, default=10, required=False)
parser.add_argument('--gpu', action="store_true", default=False, required=False)
parser.add_argument('--dropout', default=True, action="store_true", required=False)
parser.add_argument('--dropout_ratio', type=float, default=0.3, required=False)


class Config:
    def __init__(self, args):
        self.train_data, self.test_data, self.valid_data, self.mapping = load_data(args.data_dir)
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        use_gpu = args.gpu
        hidden_units = args.hidden_units
        self.architecture = args.arch
        use_dropout = args.dropout
        dropout_ratio = args.dropout_ratio

        base_model = load_base_model(self.architecture)

        features_in = get_classifier_inputs_number(base_model)

        classifier = Classifier(features_in=features_in,
                                hidden_units=hidden_units,
                                use_dropout=use_dropout,
                                dropout_ratio=dropout_ratio)

        self.save_dir = args.save_dir
        self.model = get_model(base_model, classifier)

        cuda_is_available = torch.cuda.is_available()
        if not cuda_is_available:
            print("Cuda is not available. Only the CPU will be used")
        self.device = torch.device("cuda:0" if cuda_is_available and use_gpu else "cpu")


def train(config):
    model = config.model
    device = config.device
    model.to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr=config.learning_rate)
    criterion = nn.NLLLoss()
    train_losses, validation_losses = [], []

    with active_session():

        for epoch in range(config.epochs):
            training_loss = 0
            for images, labels in config.train_data:
                images, labels = images.to(device), labels.to(device)
                # Initialize the gradient before training step
                optimizer.zero_grad()
                logits = model.forward(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()

            # Make validation after each epoch
            # Put the model in evalution mode
            model.eval()
            valid_loss = 0
            valid_accuracy = 0

            with torch.no_grad():
                for images, labels in config.valid_data:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    valid_loss += criterion(logps, labels).item()

                    probabilities = torch.exp(logps)
                    top_probability, top_class = probabilities.topk(1, dim=1)
                    correct = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(correct.type(torch.FloatTensor)).item()

                mean_training_loss = training_loss / len(config.train_data)
                mean_valid_loss = valid_loss / len(config.valid_data)
                mean_valid_accuracy = valid_accuracy / len(config.valid_data)

                train_losses.append(mean_training_loss)
                validation_losses.append(mean_valid_loss)

                print("Epoch : {}/{}".format(epoch + 1, config.epochs),
                      "Training loss : {:.3f}".format(mean_training_loss),
                      "Validation loss : {:.3f}".format(mean_valid_loss),
                      "Validation accuracy : {:.3f} %".format(mean_valid_accuracy * 100)
                      )

                # Put model back in training mode
                model.train()

    save_model(save_directory=config.save_dir,
               arch=config.architecture,
               model=model,
               classifier=model.classifier,
               optimizer=optimizer, epoch=config.epochs,
               mapping=config.mapping)


if __name__ == "__main__":
    arguments = parser.parse_args()
    print(arguments)
    configuration = Config(arguments)
    train(configuration)
