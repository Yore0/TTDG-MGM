from detectron2.evaluation import DatasetEvaluator
from collections import defaultdict
import numpy as np
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch
from pycocotools import mask as mask_util
from scipy import ndimage
from scipy.spatial.distance import cdist


# Assuming enhanced_align and Structure_measure are already implemented or imported

class DiceEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, thres):
        self.dataset_name = dataset_name
        self.dataset_dicts = DatasetCatalog.get(self.dataset_name)
        self.score_threshold = thres

    def reset(self):
        self.dice_scores = []
        self.ea_scores = []  # Enhanced Alignment metric scores
        self.sm_scores = []  # Structural Similarity metric scores
        self.hd95_scores = []  # HD95 metric scores

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):

            # Find the matching ground truth annotations
            for item in self.dataset_dicts:
                if item['image_id'] == input['image_id']:
                    matching_annotations = item['annotations']
                    break

            pred_masks = output["instances"].pred_masks.cpu().numpy()
            pred_classes = output["instances"].pred_classes.cpu().numpy()
            pred_scores = output["instances"].scores.cpu().numpy()

            # Filter out predictions with low confidence scores
            high_conf_indices = pred_scores >= self.score_threshold
            pred_masks = pred_masks[high_conf_indices]
            pred_classes = pred_classes[high_conf_indices]

            # Get ground truth masks and class IDs
            gt_masks = [self.convert_to_binary_mask(ann["segmentation"], input["height"], input["width"]) 
                        for ann in matching_annotations]
            gt_classes = [ann["category_id"] for ann in matching_annotations]

            # Match prediction masks with ground truth masks and calculate metrics
            for pred_class, pred_mask in zip(pred_classes, pred_masks):
                best_dice_score = 0
                best_ea_score = 0
                best_sm_score = 0
                best_hd95_score = float('inf')  # Initialize with a large value
                for gt_class, gt_mask in zip(gt_classes, gt_masks):
                    if pred_class == gt_class:
                        # Dice coefficient
                        intersection = np.logical_and(pred_mask, gt_mask).sum()
                        dice_score = 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-6)
                        best_dice_score = max(best_dice_score, dice_score)

                        # Enhanced Alignment (EA) metric
                        ea_score = enhanced_align(pred_mask, gt_mask)
                        best_ea_score = max(best_ea_score, ea_score)

                        # Structural Similarity (SM) metric
                        sm_metric = Structure_measure()
                        sm_score = sm_metric.get_score(pred_mask, gt_mask)
                        best_sm_score = max(best_sm_score, sm_score)

                        # HD95
                        # hd95_score = compute_hd95(pred_mask, gt_mask)
                        # best_hd95_score = min(best_hd95_score, hd95_score)

                # Save the best scores for this prediction
                self.dice_scores.append(best_dice_score * 100)
                self.ea_scores.append(best_ea_score * 100)  # Scale to percentage
                self.sm_scores.append(best_sm_score * 100)  # Scale to percentage
                # self.hd95_scores.append(best_hd95_score)  # HD95 doesn't need scaling

    def evaluate(self):
        # Calculate the mean of each metric
        mean_dice = np.mean(self.dice_scores)
        mean_ea = np.mean(self.ea_scores)
        mean_sm = np.mean(self.sm_scores)
        # mean_hd95 = np.mean(self.hd95_scores)

        return {
            "Dice Coefficient": mean_dice,
            "Enhanced Alignment Metric": mean_ea,
            "Structural Similarity Metric": mean_sm,
            # "HD95": mean_hd95
        }
    
    def convert_to_binary_mask(self, segmentation, height, width):
        if isinstance(segmentation, list):
            # If it's a polygon format
            rles = mask_util.frPyObjects(segmentation, height, width)
            rle = mask_util.merge(rles)
        elif isinstance(segmentation['counts'], list):
            # If it's uncompressed RLE format
            rle = mask_util.frPyObjects([segmentation], height, width)[0]
        else:
            # If it's compressed RLE format
            rle = segmentation
        
        mask = mask_util.decode(rle).astype(np.bool)
        return mask

# Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
def enhanced_align(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)

    th = 2 * pred.mean()
    if th > 1:
        th = 1
    FM = np.zeros(gt.shape)
    FM[pred >= th] = 1
    FM = np.array(FM, dtype=bool)
    GT = np.array(gt, dtype=bool)
    dFM = np.double(FM)
    if (sum(sum(np.double(GT))) == 0).all():
        enhanced_matrix = 1.0-dFM
    elif (sum(sum(np.double(~GT))) == 0).all():
        enhanced_matrix = dFM
    else:
        dGT = np.double(GT)
        align_matrix = AlignmentTerm(dFM, dGT)
        enhanced_matrix = EnhancedAlignmentTerm(align_matrix)
    [w, h] = np.shape(GT)
    score = sum(sum(enhanced_matrix)) / (w * h - 1 + 1e-8)
    return score

def AlignmentTerm(dFM, dGT):
    mu_FM = np.mean(dFM)
    mu_GT = np.mean(dGT)
    align_FM = dFM - mu_FM
    align_GT = dGT - mu_GT
    align_Matrix = 2. * (align_GT * align_FM) / (align_GT * align_GT + align_FM * align_FM + 1e-8)
    return align_Matrix

def EnhancedAlignmentTerm(align_Matrix):
    enhanced = np.power(align_Matrix + 1, 2) / 4
    return enhanced


class Structure_measure():
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def get_score(self, pred, gt):
        pred = np.array(pred)
        gt = np.array(gt)

        gt = gt > 0.5
        score = self.cal(pred, gt)
        return score

    def cal(self, pred, gt):
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
        return score

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, np.logical_not(gt))

    def s_object(self, in1, in2):
        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2 * x / (pow(x, 2) + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        [y, x] = ndimage.center_of_mass(gt)
        y = int(round(y)) + 1
        x = int(round(x)) + 1
        [gt1, gt2, gt3, gt4, w1, w2, w3, w4] = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)

        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h * w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        LT = pred[0:y, 0:x]
        RT = pred[0:y, x:w]
        LB = pred[y:h, 0:x]
        RB = pred[y:h, x:w]

        return LT, RT, LB, RB

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = h * w

        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1 - x) * (in2 - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score
    
# def compute_hd95(pred_mask, gt_mask):
#     """
#     计算 HD95 (95th Percentile Hausdorff Distance)。
#     """
#     pred_points = np.argwhere(pred_mask)  # 获取预测掩码的坐标点
#     gt_points = np.argwhere(gt_mask)      # 获取真实掩码的坐标点

#     if pred_points.size == 0 or gt_points.size == 0:
#         return float('inf')  # 如果任意掩码为空，则返回无穷大

#     # 计算距离矩阵
#     distances = cdist(pred_points, gt_points, metric='euclidean')

#     # 获取所有最小距离
#     min_distances_pred_to_gt = distances.min(axis=1)
#     min_distances_gt_to_pred = distances.min(axis=0)

#     # 合并所有最小距离并计算 95 百分位数
#     all_min_distances = np.concatenate([min_distances_pred_to_gt, min_distances_gt_to_pred])
#     hd95 = np.percentile(all_min_distances, 95)

#     return hd95