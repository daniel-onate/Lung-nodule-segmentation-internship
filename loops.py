import time
import torch
from matplotlib import pyplot as plt
import utils

class Trainer():

    def __init__(self, model, device, criterion, optimizer, num_epochs, early_stopping, train_loader, val_loader, test_loader, save_path):

        self.start = 0
        self.end = 0
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_path = save_path


    def train(self):

        start = time.time()

        self.model.to(self.device)

        train_loss_list = []
        val_loss_list = []

        #training loop
        for epoch in range(self.num_epochs):

            epoch_start = time.time()

            #training
            train_loss = 0.0
            self.model.train()

            for images, masks in self.train_loader:

                images, masks = images.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

                #balance the loss per batch size
                train_loss += loss.item() * images.size(0)
            
            train_loss /= len(self.train_loader.dataset)

            train_loss_list.append(train_loss)

            #validation
            val_loss = 0.0
            self.model.eval()
            
            with torch.no_grad():

                for images, masks in self.val_loader:

                    images, masks = images.to(self.device), masks.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                    #balance the loss per batch size
                    val_loss += loss.item() * images.size(0)
                
            val_loss /= len(self.val_loader.dataset)

            val_loss_list.append(val_loss)

            self.early_stopping.check(val_loss)
            if self.early_stopping.stop_training:
                break
            
            epoch_end = time.time()

            print(f"Epoch [{epoch+1}/{self.num_epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}   Runtime: {epoch_end - epoch_start}")


        end = time.time()

        print(f"Train runtime: {end - start}")

        #plot training and validation loss
        plt.figure()
        plt.plot(train_loss_list, label='training loss')
        plt.plot(val_loss_list,label='validation loss')
        plt.title('Traning and validation loss')
        plt.legend()
        plt.show
        plt.savefig(self.save_path)
        print(self.save_path)


    def test(self):

        start = time.time()

        test_dice = 0.0
        test_loss = 0.0
        self.model.eval()

        with torch.no_grad():

            for images, masks in self.test_loader:

                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                dice = utils.dice_coeff(outputs, masks)
                loss = self.criterion(outputs, masks)
                
                #balance the loss per batch size
                test_dice += dice * images.size(0)
                test_loss += loss.item() * images.size(0)
            
        test_dice /= len(self.test_loader.dataset)    
        test_loss /= len(self.test_loader.dataset)

        end = time.time()

        print(f"Test runtime {end - start}")
        print(f"Mean dice coefficient on test set: {test_dice:.4f}")
        print(f"Mean loss on test set: {test_loss:.4f}")


    def get_model(self):
        return self.model