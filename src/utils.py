from torchvision import datasets, transforms, models
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_data(dataset, mean, std):
    
    logger.info(f"transforming data...")
    if dataset in ['train']:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset in ['valid', 'test']:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform

def prepare_data(directory):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_loaders = {}
    
    for i in ['train', 'valid', 'test']:
        
        logger.info(f"preparing {i} data...")
        
        path = directory + '/' + i

        dataset_obj = datasets.ImageFolder(path, transform=transform_data(i, mean, std))
        
        if i == 'train':
            data_loaders[i+'_loader'] = torch.utils.data.DataLoader(dataset_obj, batch_size=64, shuffle=True)
            class_to_idx = dataset_obj.class_to_idx
        elif i in ['valid', 'test']:
            data_loaders[i+'_loader'] = torch.utils.data.DataLoader(dataset_obj, batch_size=64)
            
    return data_loaders, class_to_idx

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    logger.info(f"processing image...")
    # Load the image using PIL
    image = Image.open(image)
    
    # Get resize shape
    width, height = image.size

    # Calculate the left, upper, right, and lower pixel coordinates of the center 224x224 portion of the image
    left = (width - 224) / 2
    upper = (height - 224) / 2
    right = left + 224
    lower = upper + 224
    
    # Crop the image using the calculated pixel coordinates
    cropped_image = image.crop((left, upper, right, lower))
    
    # Convert the image to a Numpy array
    np_image = np.array(cropped_image)
    
    # Normalize the color channels
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    np_image = np_image / 255
    np_image = (np_image - means) / stds
    
    # Reorder dimensions to match expected input of PyTorch model
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.array(image).transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def plot_graphs(probability, classes):
    
    result_df = pd.DataFrame({'probability': probability, 'classes': classes})
    result_df = result_df.sort_values('probability', ascending=False)
    sns.barplot(x='probability', y='classes', data=result_df, color='blue')
    plt.show()
