from detectron2.evaluation import DatasetEvaluator
from collections import defaultdict
import numpy as np
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch
from pycocotools import mask as mask_util


class DiceEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, thres):
        self.dataset_name = dataset_name
        self.dataset_dicts = DatasetCatalog.get(self.dataset_name)
        self.score_threshold = thres

    def reset(self):
        self.dice_scores = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):

            for item in self.dataset_dicts:
                if item['image_id'] == input['image_id']:
                    matching_annotations = item['annotations']
                    break

            pred_masks = output["instances"].pred_masks.cpu().numpy()
            pred_classes = output["instances"].pred_classes.cpu().numpy()
            pred_scores = output["instances"].scores.cpu().numpy()

            # 筛选出置信度分数高于阈值的预测结果
            high_conf_indices = pred_scores >= self.score_threshold
            pred_masks = pred_masks[high_conf_indices]
            pred_classes = pred_classes[high_conf_indices]

            # 获取真实的分割掩码和类别ID
            gt_masks = [self.convert_to_binary_mask(ann["segmentation"], input["height"], input["width"]) 
                        for ann in matching_annotations]
            gt_classes = [ann["category_id"] for ann in matching_annotations]

            # 匹配预测掩码和真实掩码
            for pred_class, pred_mask in zip(pred_classes, pred_masks):
                best_dice_score = 0
                for gt_class, gt_mask in zip(gt_classes, gt_masks):
                    if pred_class == gt_class:
                        intersection = np.logical_and(pred_mask, gt_mask).sum()
                        dice_score = 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-6)
                        best_dice_score = max(best_dice_score, dice_score)
                
                self.dice_scores.append(best_dice_score)

    def evaluate(self):
        mean_dice = np.mean(self.dice_scores)
        return {"Dice Coefficient": mean_dice}
    
    def convert_to_binary_mask(self, segmentation, height, width):
        if isinstance(segmentation, list):
            # 如果是多边形格式
            rles = mask_util.frPyObjects(segmentation, height, width)
            rle = mask_util.merge(rles)
        elif isinstance(segmentation['counts'], list):
            # 如果是未压缩的 RLE 格式
            rle = mask_util.frPyObjects([segmentation], height, width)[0]
        else:
            # 如果是压缩的 RLE 格式
            rle = segmentation
        
        mask = mask_util.decode(rle).astype(np.bool)
        return mask