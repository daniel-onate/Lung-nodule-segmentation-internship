

def apply_windowing(Input,W,L):

    min_HU=L-(0.5*W)
    max_HU=L+(0.5*W)

    Input[Input<min_HU]=min_HU
    Input[Input>max_HU]=max_HU

    return Input


def dice_coeff(outputs, masks, smooth=1e-6):

    outputs = outputs.view(-1)
    masks = masks.view(-1)

    intersection = (outputs * masks).sum()
    dice = (2. * intersection + smooth) / (outputs.sum() + masks.sum() + smooth)

    return dice.item()