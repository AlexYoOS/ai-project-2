import sys
import utils
import json

print("This is a training file")

def cat_to_name()
    
    with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    return cat_to_name


def main():
    
    directory = sys.argv[1]
    data_loaders, class_to_idx = utils.prepare_data(directory)
    
    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 4096)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(0.5)),
                            ('fc2', nn.Linear(4096, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    model.classifier = classifier

    # Set the loss function and optimizer
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
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
                for images, labels in valid_loader:
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
    
    model.class_to_idx = class_to_idx
    
    checkpoint = {'structure': 'vgg16',
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': class_to_idx,
              'optimizer_state_dict': optimizer.state_dict(),
              'epochs': epochs
              }
    torch.save(checkpoint, '.saved_models/checkpoint.pth')

if __name__ == '__main__':
    main()