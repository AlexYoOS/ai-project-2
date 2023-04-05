import sys
import utils
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cat_to_name():
    
    logger.info(f"reading cat_to_name...")
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def main():
    
    data_dir = args.data_dir
    hidden_units = args.hidden_units
    learning_rate = args.learning_rate
    epochs = args.epochs
    use_gpu = args.gpu
    
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"using GPU for training...")
    else:
        device = torch.device('cpu')
        logger.info(f"using CPU for training...")

    logger.info(f"reading images from {data_dir}...")
    
    data_loaders, class_to_idx = utils.prepare_data(data_dir)
    
    if args.architecture == 'vgg16':
        logger.info(f"selecting vgg16 model architecture...")
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False  
    elif args.architecture == 'resnet':
        logger.info(f"selecting resnet model architecture...")
        model = models.resnet50(pretrained=True)
      
        for param in model.parameters():
            param.requires_grad = True
    else:
        print('Invalid model architecture, exiting...')
        sys.exit(1)
    
    logger.info(f"defining classifier...")
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 4096)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(0.5)),
                            ('fc2', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    model.classifier = classifier
    
    # Move the model to the selected device
    model.to(device)

    # Set the loss function and optimizer
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    logger.info(f"running training...")
    epochs = epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        logger.info(f"epoch {epoch}...")
        train_loss = 0
        for images, labels in data_loaders['train_loader']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        else:
            valid_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                for images, labels in data_loaders['valid_loader']:
                    logger.info(f"performing validation for epoch {epoch}...")
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    loss = criterion(logps, labels)
                    valid_loss += loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            model.train()
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                "Training Loss: {:.3f}.. ".format(train_loss/len(data_loaders['train_loader'])),
                "Validation Loss: {:.3f}.. ".format(valid_loss/len(data_loaders['valid_loader'])),
                "Validation Accuracy: {:.3f}".format(accuracy/len(data_loaders['valid_loader'])))
    
    logger.info(f"measuring accuracy on test data...")
    test_loss = 0
    test_accuracy = 0
    model.eval()

    with torch.no_grad():
        model.eval()
        for images, labels in data_loaders['test_loader']:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    test_loss /= len(data_loaders['test_loader'])
    test_accuracy /= len(data_loaders['test_loader'])
    print("Test Loss: {:.3f}.. ".format(test_loss),
          "Test Accuracy: {:.3f}".format(test_accuracy))
    
    model.class_to_idx = class_to_idx
    
    logger.info(f"defining {args.architecture} checkpoint...")
    checkpoint = {'structure': args.architecture,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': class_to_idx,
              'optimizer_state_dict': optimizer.state_dict(),
              'epochs': epochs
              }
    
    logger.info(f"saving the {args.architecture} model...")
    torch.save(checkpoint, './saved_models/checkpoint.pth')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Flower classifier training')
    parser.add_argument('data_dir', type=str, help='path to the image directory')
    parser.add_argument('--architecture', type=str, default='vgg16', help='model architecture (vgg16 or resnet)')
    parser.add_argument('--hidden_units', type=int, default=4096, help='number of hidden units in the new classifier')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for the classifier')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    args = parser.parse_args()
    
    main()