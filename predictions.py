import os
import pickle
import numpy as np
import pandas as pd
import operator

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random
#import matplotlib.pyplot as plt
import skimage.io as io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg 
from detectron2.utils.visualizer import Visualizer

from PIL import Image

def pred_price(image_path):

    def DICE_COE(mask1, mask2):
        intersect = np.sum(mask1*mask2)
        fsum = np.sum(mask1)
        ssum = np.sum(mask2)
        dice = (2 * intersect ) / (fsum + ssum)
        dice = np.mean(dice)
        #dice = round(dice, 3) # for easy reading
        return dice  

    #damage_type model

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this  dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (damage)
    cfg.MODEL.RETINANET.NUM_CLASSES = 3 # only has one class (damage)

    cfg.MODEL.DEVICE = "cpu"

    cfg.MODEL.WEIGHTS = os.path.join('models', "model_final_1.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.57  # set a custom testing threshold for this model
    #cfg.DATASETS.TEST = ("damage_type_val")
    predictor1 = DefaultPredictor(cfg)

    metadata = {'thing_classes':['minor', 'moderate', 'severe']}
    im = io.imread(image_path)
    outputs = predictor1(im)
   
    v = Visualizer(im[:, :, ::-1],
                    metadata=metadata,
                    scale=0.5, 
                    #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    ima = Image.fromarray(out.get_image()[:, :, ::-1])
    
    save_path = 'pred_'+image_path.split('\\')[1]
    ima.save('static/'+save_path)

    #car_part model
    cfg_2 = get_cfg()
    cfg_2.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg_2.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this  dataset (default: 512)
    cfg_2.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (damage)
    cfg_2.MODEL.RETINANET.NUM_CLASSES = 5 # only has one class (damage)

    cfg_2.MODEL.DEVICE = "cpu"

    cfg_2.MODEL.WEIGHTS = os.path.join('models', "model_final_2.pth")
    cfg_2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.70  # set a custom testing threshold for this model
    #cfg.DATASETS.TEST = ("car_part_val")
    predictor2 = DefaultPredictor(cfg_2)


    #loading train damage dataset
    with open('repair_cost_dataset', 'rb') as dataset:
 
        repair_cost_dataset = pickle.load(dataset)

    #creating damage features

    dict = {'headlamp_dice':[],'rear_bumper_dice':[],'door_dice':[],'hood_dice':[],'front_bumper_dice':[],'minor':[],'moderate':[],'severe':[]}

    damage_categories = ['minor','moderate','severe']
    part_categories = ['headlamp','rear_bumper','door','hood','front_bumper']
    #coco = COCO(val_json)
    #image_ids = np.arange(11)
    img = io.imread(image_path)
    damage_type_outputs = predictor1(img)
    car_part_outputs = predictor2(img)

    cat_ids1 = [0,1,2]
    anns = damage_type_outputs['instances'].pred_masks.cpu().numpy()

    cat_ids2 = [0,1,2,3,4]
    anns2 = car_part_outputs['instances'].pred_masks.cpu().numpy()

    #print(len(anns2))
    # break 
    global headlamp_dice
    global rear_bumper_dice
    global door_dice
    global hood_dice
    global front_bumper_dice

    for j in range(len(anns)):


        headlamp_dice = 0
        rear_bumper_dice = 0
        door_dice = 0
        hood_dice = 0
        front_bumper_dice = 0
        cats = []
        
        mask1 = anns[j]

        for k in range(len(anns2)):
            mask2 = anns2[k]
            dice_coe = DICE_COE(mask1,mask2)
            part_category_id = int(car_part_outputs['instances'].pred_classes[k])
            part_affected = part_categories[part_category_id]
            cats.append(int(car_part_outputs['instances'].pred_classes[k]))
            globals()[part_affected+'_dice'] += dice_coe
        
        for k in cat_ids2:
            part_name = part_categories[k]
            dict[part_name+'_dice'].append(globals()[part_name+'_dice'])

        damage_type = damage_categories[int(damage_type_outputs['instances'].pred_classes[j])]
        dict[damage_type].append(1)

        for k in cat_ids1:
            if k != int(damage_type_outputs['instances'].pred_classes[j]):
                damage_type = damage_categories[k]
                dict[damage_type].append(0)
    
    val_repair_cost_dataset = pd.DataFrame(dict)

    unknown = []
    for i in val_repair_cost_dataset.iloc:
        if i['headlamp_dice'] == i['rear_bumper_dice'] == i['door_dice'] == i['hood_dice'] == i['front_bumper_dice'] == 0:
            unknown.append(1)
        else:
            unknown.append(0)

    val_repair_cost_dataset.insert(loc=5, column='unknown', value=unknown)

    #calculate total_price

    operatorlookup = {
        '+': operator.add,
        '-': operator.sub,
        '==': operator.eq,
        '>': operator.gt
    }

    total_price = 0

    for i in val_repair_cost_dataset.iloc:

        d = i.to_dict()

        a = []
        opt = []
        for i in d.values():
            if i > 0 and i!=1:
                a.append(0)
                opt.append('>')
            elif i == 1:
                a.append(1)
                opt.append('==')
            else:
                a.append(0)
                opt.append('==')

        price = repair_cost_dataset[(operatorlookup[opt[0]](repair_cost_dataset[val_repair_cost_dataset.columns[0]], a[0])) & 
                            (operatorlookup[opt[1]](repair_cost_dataset[val_repair_cost_dataset.columns[1]], a[1])) &
                            (operatorlookup[opt[2]](repair_cost_dataset[val_repair_cost_dataset.columns[2]], a[2])) &
                            (operatorlookup[opt[3]](repair_cost_dataset[val_repair_cost_dataset.columns[3]], a[3])) &
                            (operatorlookup[opt[4]](repair_cost_dataset[val_repair_cost_dataset.columns[4]], a[4])) &
                            (operatorlookup[opt[5]](repair_cost_dataset[val_repair_cost_dataset.columns[5]], a[5])) &
                            (operatorlookup[opt[6]](repair_cost_dataset[val_repair_cost_dataset.columns[6]], a[6])) &
                            (operatorlookup[opt[7]](repair_cost_dataset[val_repair_cost_dataset.columns[7]], a[7])) &
                            (operatorlookup[opt[8]](repair_cost_dataset[val_repair_cost_dataset.columns[8]], a[8]))].price.mean()

        total_price += price

    return total_price, save_path


if  __name__ == '__main__':
    price,path = pred_price('static\9.jpg')
    print(price)