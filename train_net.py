#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
import adapteacher.data.datasets.builtin

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ateacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            if cfg.TEST.EVAL_STU:
                res = Trainer.test(cfg, ensem_ts_model.modelStudent)
            else:
                res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
            
        else:
            model = Trainer.build_model(cfg)
            # total_params = sum(p.numel() for p in model.parameters())
            # print(f"Total Parameters: {total_params}")
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model, Trainer.build_optimizer(cfg, model))
        # if cfg.TEST.DICE:
        #     resl = list(res.values())
        #     print(f"bbox---{resl[0]['bbox']}")
        #     print(f"segm---{resl[0]['segm']}")
        #     print(resl[1])
        #     with open(os.path.join(cfg.OUTPUT_DIR, 'result_ap.txt'), 'a') as file:
        #         file.write('loading data from: ' + cfg.MODEL.WEIGHTS + "\n")
        #         file.write(json.dumps(resl[0]['bbox']) + "\n")
        #         file.write(json.dumps(resl[0]['segm']) + "\n")
        #         file.write(json.dumps(resl[1]) + "\n")
        # else:
        print(res)
        with open(os.path.join(cfg.OUTPUT_DIR, 'result_ap.txt'), 'a') as file:
            file.write('loading data from: ' + cfg.MODEL.WEIGHTS + "\n")
            file.write(json.dumps(res) + "\n")
        
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.resume = True
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
