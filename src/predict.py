import sys
import torch
from torchvision import datasets, transforms, models
from torch.nn.functional import softmax
import utils
import argparse
import logging
import json
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_checkpoints(file):
    
    checkpoint = torch.load(file)
    structure = checkpoint['structure']
    
    if checkpoint['structure'].lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['structure'].lower() == 'resnet':
        model = models.resnet50(pretrained=True)
    else:
        print("Unsupported structure type: {}".format(structure))
        return None, None, None
        
    for param in model.parameters():
        param.requires_grad = False
            
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def main():
    
    input_image = args.input_image
    checkpoint_file = args.checkpoint_file
    category_names_file = args.category_names_file
    topk = args.topk
    use_gpu = args.gpu
    
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"using GPU for predict...")
    else:
        device = torch.device('cpu')
        logger.info(f"using CPU for predict...")

    model = load_checkpoints(checkpoint_file)    
    
    # Move the model to the selected device
    model.to(device)

    # Load the image
    image = Image.open(input_image)

    # Transform and convert to tensor
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
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
        
        top_indices = topk_classes.cpu().numpy().tolist()
        top_indices = top_indices[0]
        idx_to_class = {x:y for y, x in model.class_to_idx.items()}
        topk_classes = [idx_to_class[x] for x in top_indices]      

        # Convert the class indices to class names
        with open(category_names_file, 'r') as f:
            cat_to_name = json.load(f)
        topk_classes = [cat_to_name[str(idx_to_class[x])] for x in top_indices]
        
        print(topk_classes[:topk])
        print(topk_probabilities[0].tolist()[:topk])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Flower classifier predict')
    parser.add_argument('input_image', type=str, help='path to the image')
    parser.add_argument('checkpoint_file', type=str, help='path to the checkpoint')
    parser.add_argument('category_names_file', type=str, help='path to the category name')
    parser.add_argument('--topk', type=int, default=5, help='number of top k matches')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    args = parser.parse_args()
    
    main()