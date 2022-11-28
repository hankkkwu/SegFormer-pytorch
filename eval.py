import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from segformer import segformer_mit_b3
from utils import meanIoU, get_BDD_datasets, get_dataloaders, visualize_predictions, preprocess
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp

from collections import namedtuple

#################
# Drivable area #
#################
# Each label is a tuple with name, class id and color
Label = namedtuple( "Label", [ "name", "train_id", "color"])
drivables = [ 
             Label("direct", 0, (215, 40, 135)),        # red
             Label("alternative", 1, (61, 143, 86)),  # cyan
             Label("background", 2, (0, 0, 0)),        # black          
            ]

drivable_train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
drivable_train_id_to_color = np.array(drivable_train_id_to_color)


#########################
# semantic segmentation #
#########################
# Constants for Standard color mapping
# Based on https://github.com/mcordts/cityscapesScripts
cs_labels = namedtuple('CityscapesClass', ['name', 'train_id', 'color'])
cs_classes = [
    cs_labels('road',          0, (128, 64, 128)),
    cs_labels('sidewalk',      1, (244, 35, 232)),
    cs_labels('building',      2, (70, 70, 70)),
    cs_labels('wall',          3, (102, 102, 156)),
    cs_labels('fence',         4, (190, 153, 153)),
    cs_labels('pole',          5, (153, 153, 153)),    
    cs_labels('traffic light', 6, (250, 170, 30)),
    cs_labels('traffic sign',  7, (220, 220, 0)),
    cs_labels('vegetation',    8, (107, 142, 35)),
    cs_labels('terrain',       9, (152, 251, 152)),
    cs_labels('sky',          10, (70, 130, 180)),
    cs_labels('person',       11, (220, 20, 60)),
    cs_labels('rider',        12, (255, 0, 0)),
    cs_labels('car',          13, (0, 0, 142)),
    cs_labels('truck',        14, (0, 0, 70)),
    cs_labels('bus',          15, (0, 60, 100)),
    cs_labels('train',        16, (0, 80, 100)),
    cs_labels('motorcycle',   17, (0, 0, 230)),
    cs_labels('bicycle',      18, (119, 11, 32)),
    cs_labels('ignore_class', 19, (0, 0, 0)),
]

semseg_train_id_to_color = [c.color for c in cs_classes if (c.train_id != -1 and c.train_id != 255)]
semseg_train_id_to_color = np.array(semseg_train_id_to_color)


def evaluate_model(model, dataloader, criterion, metric_class, num_classes, device):
    '''evaluate model on dataset'''
    model.eval()
    total_loss = 0.0
    metric_object = metric_class(num_classes)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)                
            y_preds = model(inputs)

            # calculate loss
            loss = criterion(y_preds, labels)
            total_loss += loss.item()

            # update batch metric information            
            metric_object.update(y_preds.cpu().detach(), labels.cpu().detach())

    evaluation_loss = total_loss / len(dataloader)
    evaluation_metric = metric_object.compute()
    return evaluation_loss, evaluation_metric

def predict_video(model, model_name, input_video_path, output_dir, 
                target_width, target_height, device, task):
    file_name = input_video_path.split(os.sep)[-1].split('.')[0]
    output_filename = f'{file_name}_{model_name}_output.avi'
    output_video_path = os.path.join(output_dir, *[output_filename])
    print(output_video_path)

    # handles for input output videos
    input_handle = cv2.VideoCapture(input_video_path)
    output_handle = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), \
                                    20, (target_width, target_height))

    # create progress bar
    num_frames = int(input_handle.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total = num_frames, position=0, leave=True)

    while(input_handle.isOpened()):
        ret, frame = input_handle.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # create torch tensor to give as input to model
            pt_image = preprocess(frame)
            pt_image = pt_image.to(device)

            # get model prediction and convert to corresponding color
            pred = model(pt_image.unsqueeze(0))     # pred = [N, C, H, W]
            # print('semantic segmentation output: ', pred.shape)
            y_pred = torch.argmax(pred, dim=1).squeeze(0)
            predicted_labels = y_pred.cpu().detach().numpy()
            if task == 'drivable':
                cm_labels = (drivable_train_id_to_color[predicted_labels]).astype(np.uint8)
            elif task == 'semseg':
                cm_labels = (semseg_train_id_to_color[predicted_labels]).astype(np.uint8)
            
            # get the location of a sepecific class
            pos_y, pos_x = np.where(predicted_labels==8)

            # overlay prediction over input frame
            overlay_image = cv2.addWeighted(frame, 1, cm_labels, 0.25, 0)
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)

            # write output result and update progress
            output_handle.write(overlay_image)
            pbar.update(1)

        else:
            break

    output_handle.release()
    input_handle.release()


if __name__ == '__main__':
    """ Evaluate : Evaluate the model on Test Data and visualize results """
    NUM_CLASSES = 19    # 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # criterion = smp.losses.DiceLoss('multiclass', classes=[0,1,2], log_loss = True, smooth=1.0)
    criterion = nn.CrossEntropyLoss()

    # load the model
    model = segformer_mit_b3(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load('trained_model/BDD100K/segformer_mit_b3_bdd_19CLS_CE_loss.pt', map_location=device))

    """
    # Evaluate on test dataset
    _, test_metric = evaluate_model(model, test_dataloader, criterion, meanIoU, NUM_CLASSES, device)
    print(f"\nModel has {test_metric} mean IoU in test set")

    # Visualize the result
    num_test_samples = 2
    _, axes = plt.subplots(num_test_samples, 3, figsize=(3*6, num_test_samples * 4))
    visualize_predictions(model, test_set, axes, device, numTestSamples=num_test_samples, 
                        id_to_color = train_id_to_color)
    """

    predict_video(model, 'segformer_mit_b3_bdd_3CLS_CEloss_4Epoch.pt', 
                     input_video_path = 'dataset/demoVideo/stuttgart_1024_512.avi', 
                     output_dir = 'results', 
                     target_width = 1024, 
                     target_height = 512, 
                     device = device,
                     task = 'semseg')   # drivable, semseg
