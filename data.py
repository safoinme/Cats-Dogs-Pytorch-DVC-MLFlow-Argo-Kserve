import torch
from torchvision import datasets, transforms
import parameters


class DataModule():

    def __init__(self, **kwargs):
        self.Normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.TrainTransform = None
        self.ValidationTransform = None
        self.TrainImageFolder = None
        self.ValidationImageFolder = None
        self.TrainLoader = None
        self.ValidationLoader = None
        self.Classes = None

    def createTrainTransform(self):
        self.TrainTransform = transforms.Compose([
            transforms.Resize((parameters.IMG_HEIGHT, parameters.IMG_WIDTH)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.Normalize])
        return self.TrainTransform

    def createValidationTransform(self):
        self.ValidationTransform = transforms.Compose([
            transforms.Resize((parameters.IMG_HEIGHT, parameters.IMG_WIDTH)),
            transforms.ToTensor(),
            self.Normalize])
        return self.ValidationTransform

    def createTrainImageFolder(self):
        self.TrainImageFolder = datasets.ImageFolder(parameters.DATASET_HOME +
                                                     'train', self.createTrainTransform())
        return self.TrainImageFolder

    def createValidationImageFolder(self):
        self.ValidationImageFolder = datasets.ImageFolder(parameters.DATASET_HOME +
                                                          'validation', self.createValidationTransform())
        return self.ValidationImageFolder

    def createTrainDataLoader(self):
        self.TrainLoader = torch.utils.data.DataLoader(self.createTrainImageFolder(),
                                batch_size=parameters.BATCH_SIZE,
                                shuffle=parameters.SHUFFLING,
                                num_workers=parameters.NUM_WORKERS)
        return self.TrainLoader


    def createValidationDataLoader(self):
        self.ValidationLoader = torch.utils.data.DataLoader(self.createValidationImageFolder(),
                                batch_size=parameters.BATCH_SIZE,
                                shuffle=parameters.SHUFFLING,
                                num_workers=parameters.NUM_WORKERS)
        return self.ValidationLoader


    def getDatasetClasses(self):
        self.Classes = self.TrainImageFolder.classes
        return self.Classes
