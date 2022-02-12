from model import *
from data import *
import parameters
import torch.optim as optim
from tqdm import tqdm

def train(model,train_loader): 
    optimizer = optim.SGD(model.parameters(), lr=parameters.LEARNING_RATE, momentum=parameters.MOMENTUM)
    criterion = nn.CrossEntropyLoss()
    print('Start Training')
    for epoch in range(parameters.NUM_EPOCHS):
        running_loss = 0.0
        loop = tqdm(train_loader, leave=True)
        #print("Epoch ==>",epoch,"/",epochs)
        for _,data in enumerate(train_loader, 0):
         
            inputs, labels = data
            optimizer.zero_grad()
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            loop.set_postfix(running_loss=loss.item())
        print("val loss : ",running_loss)
        
    print('Finished Training')

def validation(model,val_loader,classes):
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(2):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

def main():
  model = resnet18()
  Dataset = DataModule()
  train_loader = Dataset.createTrainDataLoader()
  val_loader = Dataset.createValidationDataLoader()
  classes = Dataset.getDatasetClasses()
  train(model,train_loader)
  validation(model,val_loader,classes)
  return model

if __name__ == "__main__":
    main()