def apply_windowing(Input,W,L):

    min_HU=L-(0.5*W)
    max_HU=L+(0.5*W)

    Input[Input<min_HU]=min_HU
    Input[Input>max_HU]=max_HU

    return Input


def dice_coeff(outputs, masks, smooth=1e-6):

    outputs = outputs.view(-1)
    outputs = (outputs > 0.5).float()
    masks = masks.view(-1)

    intersection = (outputs * masks).sum()
    dice = (2. * intersection + smooth) / (outputs.sum() + masks.sum() + smooth)

    return dice.item()

def precision(outputs, masks):

    outputs = outputs.view(-1)
    outputs = (outputs > 0.5).float()
    masks = masks.view(-1)

    TP = ((outputs == 1) & (masks == 1)).sum().item()
    FP = ((outputs == 1) & (masks == 0)).sum().item()
    
    precision = TP / (TP + FP + 1e-8)

    return precision

def recall(outputs, masks):

    outputs = outputs.view(-1)
    outputs = (outputs > 0.5).float()
    masks = masks.view(-1)

    TP = ((outputs == 1) & (masks == 1)).sum().item()
    FN = ((outputs == 0) & (masks == 1)).sum().item()
    
    recall = TP / (TP + FN + 1e-8)

    return recall

class EarlyStopping():
    def __init__(self, patience, delta):
        self.stop_training = False
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None

    def check(self, val_loss):
        if self.best_loss == None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
                print('Early stopping')


import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, masks):

        outputs = outputs.view(-1)
        masks = masks.view(-1)

        intersection = (outputs * masks).sum()
        loss = (2. * intersection + self.smooth) / (outputs.sum() + masks.sum() + self.smooth)

        return 1 - loss
    

class ComboLoss(nn.Module):

    def __init__(self, alpha=0.5):

        super().__init__()
        self.dice_w = alpha
        self.bce_w = 1 - alpha
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, outputs, masks):

        bce_loss = self.bce(outputs, masks)
        dice_loss = self.dice(outputs, masks)

        loss = bce_loss * self.bce_w + dice_loss * self.dice_w

        return loss