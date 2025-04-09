# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import cv2
import logging
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict, abc, defaultdict
from contextlib import ExitStack, contextmanager
from typing import List, Union

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
# from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.comm import get_world_size
from adapteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from adapteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from adapteacher.engine.hooks import LossEvalHook
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from adapteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from adapteacher.solver.build import build_lr_scheduler
from adapteacher.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator, DiceEvaluator
from adapteacher.modeling.GModule.graph_config import _C as graph_opt

from detectron2.utils.analysis import FlopCountAnalysis
from .probe import OpenMatchTrainerProbe
import copy

def data2boxes(data):
    boxes = []
    for i in range(len(data)):
        boxes_i = data[i]['instances'].gt_boxes.tensor
        if boxes_i.shape[0]:
            indices = i * torch.ones((boxes_i.shape[0], 1), dtype=boxes_i.dtype, device=boxes_i.device)
            boxes_i = torch.cat([indices, boxes_i], dim=1)
            boxes.append(boxes_i)
    if len(boxes):
        boxes = torch.cat(boxes, dim=0)
        return boxes
    else:
        return None


def data2labels(data):
    labels = []
    for i in range(len(data)):
        labels_i = data[i]['instances'].gt_classes
        if labels_i.shape[0]:
            labels.append(labels_i)
    labels = torch.cat(labels, dim=0)
    return labels


def locate_feature_roialign(feature_map, boxes, image_width, image_height):
    selected_features = []
    sx = feature_map.shape[3] / image_width
    sy = feature_map.shape[2] / image_height
    if len(boxes):
        boxes_level = torch.tensor(boxes, device=feature_map.device)
        boxes_level[:, 1] *= sx
        boxes_level[:, 2] *= sy
        boxes_level[:, 3] *= sx
        boxes_level[:, 4] *= sy
        selected_features_level = roi_align(feature_map, boxes_level, output_size=1, aligned=True)
        selected_features_level = torch.flatten(selected_features_level, start_dim=1)
        selected_features = selected_features_level
    else:
        selected_features = None
    return selected_features


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, weights=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (features.device
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits).to(device) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)).to(device)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        if weights is not None:
            loss = (loss.view(anchor_count, batch_size) * weights).sum() / weights.sum()
        else:
            loss = loss.view(anchor_count, batch_size).mean()

        return loss


# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised_source")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @classmethod
    def test(cls, cfg, model, optimizer=None, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            # results_i, evaluator_ = inference_on_dataset(model, data_loader, evaluator)
            if cfg.TEST.TTT:
                for idx, inputs in enumerate(data_loader):
                # total_data_time += time.perf_counter() - start_time
                    if cfg.TEST.MIN_BATCH_NUM is not None:
                        if idx >= cfg.TEST.MIN_BATCH_NUM:
                            break

                    loss, _, _, _ = model(inputs, branch='TTT')
                    if loss is None:
                        continue

                    optimizer.zero_grad()
                    loss.backward()      
                    optimizer.step()  

            dice = DiceEvaluator(dataset_name, cfg.TEST.DICE_THRES)#DiceEval(dataset_name)
            results_i, evaluator_ = inference_on_dataset(model, data_loader, dice, cfg)

            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        # if len(results) == 1:
        #     results = list(results.values())

        # calculate the mean DICE for each dataset
        # dataset_dict = defaultdict(list)
        # for key, value in results.items():
        #     dname = key.split('_')[0] 
        #     dataset_dict[dname].append(value['Dice Coefficient'])
        # for dataset, dice_list in dataset_dict.items():
        #     mean_dice = sum(dice_list) / len(dice_list)
        #     results[f'{dataset}_mean'] = {'Dice Coefficient': mean_dice}

        dataset_dict = defaultdict(lambda: defaultdict(list))

        for key, value in results.items():
            dname = key.split('_')[0] 
            
            dataset_dict[dname]['Dice Coefficient'].append(value['Dice Coefficient'])
            dataset_dict[dname]['Enhanced Alignment Metric'].append(value['Enhanced Alignment Metric'])
            dataset_dict[dname]['Structural Similarity Metric'].append(value['Structural Similarity Metric'])

        for dataset, metrics in dataset_dict.items():
            mean_dice = sum(metrics['Dice Coefficient']) / len(metrics['Dice Coefficient'])
            mean_ea = sum(metrics['Enhanced Alignment Metric']) / len(metrics['Enhanced Alignment Metric'])
            mean_sm = sum(metrics['Structural Similarity Metric']) / len(metrics['Structural Similarity Metric'])
            
            results[f'{dataset}_mean'] = {
                'Dice Coefficient': mean_dice,
                'Enhanced Alignment Metric': mean_ea,
                'Structural Similarity Metric': mean_sm
            }

        return results
    
    @classmethod
    def test_time_training(cls, model, dataloader, batch_num):
        start_time = time.perf_counter()
        losses = 0
        for idx, inputs in enumerate(dataloader):
            # total_data_time += time.perf_counter() - start_time
            if batch_num is not None:
                if idx >= batch_num:
                    break
            losses += model(inputs, branch='TTT')
        
        return losses


# Adaptive Teacher Trainer
class ATeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # Lvxg
        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # print(f"Total Parameters: {total_params}")
        # print(f"Trainable Parameters: {trainable_params}")

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks())

        # CMT configs
        # self.contrastive = cfg.SEMISUPNET.CONTRASTIVE
        # self.supconloss = SupConLoss(contrast_mode='one')
        # if cfg.MODEL.BACKBONE.NAME == 'build_vgg_backbone':
        #     self.feature_levels = ['vgg1', 'vgg2', 'vgg3', 'vgg4']
        # elif cfg.MODEL.BACKBONE.NAME == 'build_resnet_backbone':
        #     self.feature_levels = ['res2', 'res3', 'res4']
        # else:
        #     raise NotImplementedError

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume:  #and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list
    
    # def get_label_test(self, label_data):
    #     label_list = []
    #     for label_datum in label_data:
    #         if "instances" in label_datum.keys():
    #             label_list.append(label_datum["instances"])

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if not self.cfg.SEMISUPNET.TTT:
            
            # input both strong and weak supervised data into model
            # Lvxg: mod-9.7: only input strong aug. for training
            # label_data_q.extend(label_data_k)
            record_dict, _, _, _, features = self.model(
                label_data_q, branch="supervised_source")
            
            # Lvxg: multi-graph matching for U
            targets = [x['instances'] for x in label_data_q]
            features = [feat[1] for feat in features.items()]

            nodes, labels = self.graph_generator(self.compute_locations(features), features, targets)
            loss_matching = self.multi_matching_sup(nodes, labels)
            record_dict.update({'loss_matching:': loss_matching})

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 0.1
                # if self.iter < 100:
                #     loss_dict[key] = record_dict[key] * 0.1 #Lvxg: d:1
            losses = sum(loss_dict.values())

        else:
            # if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
            #     # update copy the the whole model
            #     self._update_teacher_model(keep_rate=0.00)
            #     # self.model.build_discriminator()

            # elif (
            #     self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            # ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
            #     self._update_teacher_model(
            #         keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
            if (self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP) \
                    % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                     keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            ######################## For probe #################################
            # import pdb; pdb. set_trace() 
            gt_unlabel_k = self.get_label(unlabel_data_k)
            # gt_unlabel_q = self.get_label_test(unlabel_data_q)
            

            #  0. remove unlabeled data labels
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            #  1. generate the pseudo-label using teacher model
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak") # self.model_teacher

            ######################## For probe #################################
            # import pdb; pdb. set_trace() 

            # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
            # probe_metrics = ['compute_num_box']  
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
            # record_dict.update(analysis_pred)
            ######################## For probe END #################################

            #  2. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            #Process pseudo labels and thresholding
            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
            # record_dict.update(analysis_pred)

            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            # 3. add pseudo-label to unlabeled data

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q #+ unlabel_data_k #Lvxg

            # 4. input both strongly and weakly augmented labeled data into student model
            # flops = FlopCountAnalysis(self.model, (all_label_data, "supervised_source"))
            # print(f"FLOPs: {flops.total()}")
            record_all_label_data, _, _, _, features_source = self.model(
                all_label_data, branch="supervised_source"
            )
            record_dict.update(record_all_label_data)

            # 5. input strongly augmented unlabeled data into model
            record_all_unlabel_data, _, _, _, features_target = self.model(
                all_unlabel_data, branch="supervised_target"
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
            # give sign to the target data

            for i_index in range(len(unlabel_data_k)):
                for k, v in unlabel_data_k[i_index].items():
                    label_data_k[i_index][k + "_unlabeled"] = v

            record_all_domain_data, _, _, _, _ = self.model(label_data_k, branch="domain")
            record_dict.update(record_all_domain_data)

            _, middle_head_loss = \
                    self.graph_matching(None, (features_source, features_target), targets=(all_label_data, all_unlabel_data))
            record_dict.update(middle_head_loss)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif (
                        key == "loss_D_img_s" or key == "loss_D_img_t"
                    ):  # set weight for discriminator
                        # import pdb
                        # pdb.set_trace()
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = loss_dict #record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i, evaluator_ = inference_on_dataset(model, data_loader, evaluator)

            if cfg.TEST.DICE:
                dice = DiceEvaluator(dataset_name, cfg.TEST.DICE_THRES)#DiceEval(dataset_name)
                metrics, _ = inference_on_dataset(model, data_loader, dice)
                results[dataset_name+'_dice'] = metrics
            if cfg.TEST.DRAW:
                print('drawing predict bounding box!')
                draw_box(cfg, evaluator_)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]

        return results
    
    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, graph_opt.MODEL.FCOS.FPN_STRIDES[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations
    
def draw_box(cfg, evaluator, TH=0.9):

    colors = [
    (1.0, 0.0, 0.0, 1.0),  # 红色
    (0.0, 1.0, 0.0, 1.0),  # 绿色
    (0.0, 0.0, 1.0, 1.0),  # 蓝色
    (1.0, 1.0, 0.0, 1.0),  # 黄色
    (1.0, 0.0, 1.0, 1.0),  # 洋红色
    (0.0, 1.0, 1.0, 1.0),  # 青色
    (1.0, 0.5, 0.0, 1.0),  # 橙色
    (1.0, 1.0, 1.0, 1.0),  # white#(0.5, 0.0, 1.0, 1.0),  # 紫色
    (0.0, 0.5, 0.5, 1.0)   # 暗青色
    ]

    save_path = os.path.join(cfg.OUTPUT_DIR, 'visual')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # DatasetCatalog.register(cfg.DATASETS.TEST, your_dataset_function)
    dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    instances = evaluator._predictions

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        name = d["file_name"].split('/')[-1]
        v = Visualizer(img[:, :, ::-1], metadata=evaluator._metadata, scale=1.2)
        prediction = [item['instances'] for item in instances if item['image_id'] == d['image_id']][0]

        prediction_th = []
        for pred in prediction:
            if pred['score'] > TH:
                prediction_th.append(pred)

        for anno in prediction_th:
            bbox = anno["bbox"]
            category_id = anno["category_id"]
            color = colors[category_id - 1]
            # color_a = colors_a[category_id]

            # 转换边界框格式从 [x,y,width,height] 到 [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

            # 绘制边界框和类别标签
            v.draw_box([x1, y1, x2, y2], alpha=1 ,edge_color=color)
            # category_name = metadata.thing_classes[category_id]
            # v.draw_text(category_name, (x1, y1), color=color_a)

        result_image = v.get_output().get_image()[:, :, ::-1]
        # cv2.imwrite(f'vis_EP/public/gt_{name}', image)
        # cv2.imwrite(f'vis_EP/public/box_{name}', result_image)
        cv2.imwrite(os.path.join(save_path, name), result_image)

def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], cfg
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):

            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            
            if cfg.TEST.DRAW:
                draw_mask_img(cfg, inputs, outputs)
                mask_colors = [(1.0, 0, 0, 0.3), (0, 1.0, 0, 0.3), (0, 0, 1.0, 0.5)]
                save_path = os.path.join(cfg.OUTPUT_DIR, 'visual')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for i, output in enumerate(outputs):
                    # image = inputs[i]["image"].permute(1, 2, 0).cpu().numpy()
                    image = cv2.imread(inputs[i]["file_name"])
                    name = inputs[i]['file_name'].split('/')[-1]
                    visualizer = Visualizer(image[:, :, ::-1], metadata={}, scale=1.0)
                    # pred_masks = output["instances"].pred_masks.cpu().numpy()
                    pred_masks = output["instances"].pred_masks[output["instances"].scores > 0.8].cpu().numpy()


                    for j, mask in enumerate(pred_masks):
                        color = mask_colors[j % len(mask_colors)] 
                        visualizer.draw_binary_mask(mask, color=color, alpha=0.5)

                    vis_image = visualizer.output.get_image()[:, :, ::-1]
                    cv2.imwrite(os.path.join(save_path, name), vis_image)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results, evaluator

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def draw_mask_img(cfg, inputs, outputs):
    save_path = os.path.join(cfg.OUTPUT_DIR, 'masks')  # 保存掩码图像的路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, output in enumerate(outputs):
        # 读取原图以获取尺寸
        image = cv2.imread(inputs[i]["file_name"])
        name = inputs[i]['file_name'].split('/')[-1]
        
        # 获取预测的掩码
        pred_masks = output["instances"].pred_masks[output["instances"].scores > 0.9].cpu().numpy()
        
        # 创建一个全黑背景图像，形状与原图一致
        mask_image = np.zeros_like(image, dtype=np.uint8)

        for j, mask in enumerate(pred_masks):
            # 将单通道掩码转换为白色
            binary_mask = mask.astype(np.uint8) * 255  # 转为 0 或 255
            binary_mask_3c = cv2.merge([binary_mask] * 3)  # 扩展为三通道
            
            # 创建白色掩码
            white_mask = np.zeros_like(mask_image, dtype=np.uint8)
            white_mask[:, :, :] = binary_mask_3c  # 将掩码填充为白色
            
            # 按透明度叠加到背景
            alpha = 1  # 调整透明度
            mask_image = cv2.addWeighted(mask_image, 1, white_mask, alpha, 0)

        # 保存仅包含掩码的图像
        cv2.imwrite(os.path.join(save_path, name), mask_image)
        



