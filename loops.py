import time
import torch
from matplotlib import pyplot as plt
import utils
import random
import numpy as np

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
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
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

    def metrics(self):

        start = time.time()

        #dice standard deviation
        dice_list = []
        detected_dice_list = []
        precision_list = []
        recall_list = []
        dice_m = 0.0
        dice_sd = 0.0
        detected_dice_m = 0.0
        detected_dice_sd = 0.0
        undetected_count = 0

        idx_undetected = []
        idx_detected = []

        self.model.eval()

        with torch.no_grad():
            #unbatching all the test images and masks to calculate metrics individually
            for idx_batch, (images, masks) in enumerate(self.test_loader):

                images, masks = images.to(self.device), masks.to(self.device)

                images_list = torch.unbind(images, dim=0)
                masks_list = torch.unbind(masks, dim=0)

                for idx_inside, (unbatched_image, unbatched_masks) in enumerate(zip(images, masks)):

                    #calculating dice scores
                    #print(unbatched_image.shape, unbatched_masks.shape)
                    unbatched_output = self.model(unbatched_image.unsqueeze(0))
                    unbatched_dice = utils.dice_coeff(unbatched_output, unbatched_masks.unsqueeze(0))
                    dice_list.append(unbatched_dice)

                    #calculating detected lesion dice scores
                    if unbatched_dice < 0.05:
                        undetected_count += 1
                        idx_undetected.append([idx_batch, idx_inside])
                    if unbatched_dice >= 0.05:
                        detected_dice_list.append(unbatched_dice)
                        idx_detected.append([idx_batch, idx_inside])

                    #calculating precision and recall
                    unbatched_precision = utils.precision(unbatched_output, unbatched_masks.unsqueeze(0))
                    unbatched_recall = utils.recall(unbatched_output, unbatched_masks.unsqueeze(0))
                    precision_list.append(unbatched_precision)
                    recall_list.append(unbatched_recall)


            #plotting the images and their segmentations

            from matplotlib.patches import Patch
            import matplotlib.cm as cm

            idx_detected_sampled = random.sample(idx_detected, 3)
            #idx_undetected_sampled = random.sample(idx_undetected, 20)

            fig, axes = plt.subplots(4, 5, figsize=(20, 16))
            fig.suptitle(' 2D U-Net Detected Lesion Segmentation', fontsize=28)
            plt.subplots_adjust(top=0.94)

            

            legend_elements = [
                Patch(facecolor=cm.autumn(1), label='False Positive'),
                Patch(facecolor=cm.winter(1), label='False Negative'),
                Patch(facecolor=cm.summer(1), label='True Positive')
            ]

            axes[3, 2].legend(
                handles=legend_elements,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.05),
                ncol=5,
                frameon=True,
                prop={'size': 18},
            )

            for index, (ax) in enumerate(axes.flat):

                ax.axis('off')
                
                idx_search = idx_detected_sampled[0]

                #getting the images from the loader and plotting them
                for idx_batch, (images, masks) in enumerate(self.test_loader):

                    images, masks = images.to('cpu'), masks.to('cpu')

                    for idx_inside, (unbatched_image, unbatched_masks) in enumerate(zip(images, masks)):

                        if idx_search == [idx_batch, idx_inside]:
                            
                            #print(unbatched_image.shape)
                            ax.imshow(unbatched_image.squeeze(0), cmap='gray')
                            unbatched_output = self.model(unbatched_image.unsqueeze(0))

                            thresh_output = (unbatched_output > 0.5).squeeze([0,1])
                            union =  thresh_output * unbatched_masks.squeeze(0)

                            segment_output = np.ma.masked_where(thresh_output == 0, thresh_output)
                            ground_truth = np.ma.masked_where(unbatched_masks.squeeze(0) == 0, unbatched_masks.squeeze(0))
                            union_output = np.ma.masked_where(union == 0, union)

                            ax.imshow(segment_output, alpha=0.7, cmap='autumn')
                            ax.imshow(ground_truth, alpha=0.7, cmap='winter')
                            ax.imshow(union_output, alpha=1, cmap='summer')
                    



        #mean dice
        dice_m = sum(dice_list) / len(dice_list)
        for dice_val in dice_list:
            dice_sd += (dice_val - dice_m) ** 2
        dice_sd = (dice_sd / len(dice_list)) ** 0.5

        #mean dice for detected lesions
        detected_dice_m = sum(detected_dice_list) / len(detected_dice_list)
        for detected_dice_val in detected_dice_list:
            detected_dice_sd += (detected_dice_val - detected_dice_m) ** 2
        detected_dice_sd = (detected_dice_sd / len(detected_dice_list)) ** 0.5

        #mean precision and recall
        precision_m = sum(precision_list) / len(precision_list)
        recall_m = sum(recall_list) / len(recall_list)

        plt.figure()
        plt.hist(dice_list, bins=20)
        plt.title('3D U-Net Dice Coefficient Distribution on Test Set')
        plt.xlabel('Dice Coefficient')
        plt.ylabel('Frequency')
        plt.show()

        end = time.time()

        #print(idx_undetected, idx_detected)

        print(f"Metrics runtime {end - start}")
        print(f"Mean dice coefficient on test set: {dice_m:.4f}")
        print(f"Standard deviation of dice coefficient on test set: {dice_sd:.4f}")
        print()
        print(f"Mean precision on test set: {precision_m:.4f}")
        print(f"Mean recall on test set: {recall_m:.4f}")
        print()
        print(f"Number of undetected lesions on test set: {undetected_count}")
        print(f"Mean dice coefficient for detected lesions: {detected_dice_m:.4f}")
        print(f"Standard deviation of dice coefficient for detected lesions: {detected_dice_sd:.4f}")
        

    def get_model(self):
        return self.model