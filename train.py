from model import *
from data import *
import parameters
import torch.optim as optim
from tqdm import tqdm
import mlflow
import time
import copy
import os , json
import shutil

def mlflow_metrice(name, value, step):
    """Log a scalar value to MLflow """
    mlflow.log_metric(name, value, step)

def mlflow_parameter():
    mlflow.log_param("DEVICE", parameters.DEVICE)
    mlflow.log_param("NUM_EPOCHS", parameters.NUM_EPOCHS)
    mlflow.log_param("BATCH_SIZE", parameters.BATCH_SIZE)
    mlflow.log_param("LEARNING_RATE", parameters.LEARNING_RATE)
    mlflow.log_param("MOMENTUM", parameters.MOMENTUM)
    mlflow.log_param("NUM_WORKERS", parameters.NUM_WORKERS)
    mlflow.log_param("SHUFFLING", parameters.SHUFFLING)
    mlflow.log_param("IMG_HEIGHT", parameters.IMG_HEIGHT)
    mlflow.log_param("IMG_WIDTH", parameters.IMG_WIDTH)

def create_artifcat_folder(expirement_name):
    models_dir = "data/{}/".format(expirement_name)
    #local_dir = os.path.join(os.getcwd(), "tmp" )
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

def create_index_to_name_json(expirement_name, classes):
    indexes = {str(k): v for k, v in enumerate(classes)}
    with open("data/{}/index_to_name.json".format(expirement_name), "w+") as outfile:
        json.dump(indexes, outfile)

def copy_model_and_handler(expirement_name, handler, model):
    shutil.copy2(handler, "data/{}".format(expirement_name))
    shutil.copy2(model, "data/{}".format(expirement_name))

def create_model_flavor():
    file = open("data/save_format.txt","w+")
    file.write("pytorch")
    file.close()

def training(model, train_loader, optimizer, epoch):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader)):
        inputs = inputs.to(parameters.DEVICE)
        labels = labels.to(parameters.DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # backward + optimize only if in training phase
            # sched.step()
            loss.backward()
            optimizer.step()
            #step = epoch * len(train_loader) + batch_idx

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    train_epoch_loss = running_loss / len(train_loader.dataset) 
    #mlflow_metrice("train_loss", epoch_loss, epoch)

    train_epoch_acc = running_corrects.item() / len(train_loader.dataset)
    #mlflow_metrice("train_accuracy", epoch_acc, epoch)

    return train_epoch_loss, train_epoch_acc

def validation(model, val_loader, epoch):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    # Iterate over data.
    for batch_idx, (inputs, labels) in tqdm(enumerate(val_loader)):
        inputs = inputs.to(parameters.DEVICE)
        labels = labels.to(parameters.DEVICE)


        # forward pass
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = nn.CrossEntropyLoss()(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    val_epoch_loss = running_loss / len(val_loader.dataset) 
    #mlflow_metrice("val_loss", epoch_loss, epoch)


    val_epoch_acc = running_corrects.item() / len(val_loader.dataset)
    #mlflow_metrice("val_accuracy", epoch_acc, epoch)

    return val_epoch_loss, val_epoch_acc

def main(expirement_name):
    model_class = ImageClassifier()
    model = model_class.pretrained()
    Dataset = DataModule()
    train_loader = Dataset.createTrainDataLoader()
    val_loader = Dataset.createValidationDataLoader()
    classes = Dataset.getDatasetClasses()
    optimizer = optim.SGD(
        model.parameters(), lr=parameters.LEARNING_RATE, momentum=parameters.MOMENTUM
    )
    
    start = time.time()
    best_acc = 0.0

    for epoch in range(parameters.NUM_EPOCHS):
        print('Epoch {} Started : --------------------------------------------> '.format(epoch+1))
        
        train_loss, train_acc = training(model, train_loader, optimizer, epoch)
        mlflow_metrice("train_loss", train_loss, epoch+1)
        mlflow_metrice("train_accuracy", train_acc, epoch+1)
        
        val_loss, val_acc = validation(model, val_loader, epoch+1)
        mlflow_metrice("val_loss", val_loss, epoch+1)
        mlflow_metrice("val_accuracy", val_acc, epoch+1)
        
        print('train_loss : {}  ; train_accuracy : {}  ; val_loss : {}  ; val_accuracy : {} '.format(train_loss,train_acc,val_loss,val_acc))
        
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())       
            model_save_name = "data/{}/model.pt".format(expirement_name)
            path = F"{model_save_name}"
            torch.save(model.state_dict(), path)


    # Calculating time it took for model to train    
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    create_index_to_name_json(expirement_name, classes)
    copy_model_and_handler(expirement_name, "./handler.py", "./model.py")
    create_model_flavor()

    mlflow.log_artifact("data",artifact_path="model")

if __name__ == "__main__":
    #mlflow.mlflow.set_tracking_uri("http://127.0.0.1:5000")
    expirement_name = "cats-and-dogs"
    mlflow.set_experiment(expirement_name)
    create_artifcat_folder(expirement_name)
    with mlflow.start_run():
        mlflow_parameter()
        main(expirement_name)
