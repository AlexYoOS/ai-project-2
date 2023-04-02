import sys
import torch
from torchvision import datasets, transforms, models
from torch.nn.functional import softmax
import utils

def load_checkpoints():
    checkpoint = torch.load('saved_models/checkpoint.pth')
    
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epochs = checkpoint['epochs']
    
    return model, optimizer, epochs


def main():
    
    image_path = sys.argv[1]

    model, optimizer, epochs = load_checkpoints()    
    model.to("cpu")
    device = torch.device("cpu")

    # Load the image
    image = Image.open(image_path)

    # Transform and convert to tensor
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tensor = transform(image)

    # Use the model to predict the topk classes and their associated probabilities
    with torch.no_grad():
        output = model.forward(tensor.unsqueeze_(0).to(device))
        probabilities = softmax(output.data, dim=1).to(device)
        topk_probabilities, topk_classes = probabilities.topk(topk)

        # Convert the class indices to class names
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        topk_classes = [cat_to_name[str(class_idx.item())] for class_idx in topk_classes[0]]
        highest_probability = topk_classes[0]
    
    plt.title(highest_probability)
    plt.imshow(image)
    plt.show()
    
    utils.plot_graphs(topk_probabilities[0].tolist(), topk_classes)

    # return topk_probabilities[0].tolist(), topk_classes


if __name__ == '__main__':
    main()