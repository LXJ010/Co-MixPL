# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch
import semi
from copy import deepcopy
from typing import List, Tuple

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets import transforms as T
from util import misc


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, dataloader_semi: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    StartSemiEpoch = 0
    data_loader_semi = iter(dataloader_semi)
    _cnt = 0
    mixupnum, moasicnum = 0, 0
    print_logs = {}
    transpose = T.Compose([
        T.ToTensor(),
    ])
    t, l = 0, 0
    layers_num = 6
    coef = 2 * layers_num
    auxdevice = "cuda"
    model.fasterrcnn_ori = model.fasterrcnn_ori.to(auxdevice)
    softreg = True
    for samples, targets, imgs in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        model.train()
        samples = samples.to(device)
        target_fasterrcnn = deepcopy(targets)
        add_fasterrcnn = []
        for i in range(len(target_fasterrcnn)):
            target_fasterrcnn[i]['boxes'] = semi.rescale_bboxes(targets[i]['boxes'],
                                                                [targets[i]['size'][1], targets[i]['size'][0]])
            target_fasterrcnn[i]["labels"] = target_fasterrcnn[i]["labels"] + 1
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, memory = model(samples, targets, retmemory=True)
            else:
                outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            images: List[Tuple[int, int]] = []
            images_ori: List[Tuple[int, int]] = []
            for i in range(len(target_fasterrcnn)):
                images.append((samples.tensors.shape[-2], samples.tensors.shape[-1]))
                images_ori.append((targets[i]['size'][0].item(), targets[i]['size'][1].item()))
            target_fasterrcnn = [{k: v.to(device) for k, v in t.items()} for t in target_fasterrcnn]
            memory = model.FPN(memory)

            loss_dict_faterrcnn = model.fasterrcnn(images, original_image_sizes = images_ori, features = memory, targets = target_fasterrcnn)
            SupEncoderLoss = coef * sum(loss for loss in loss_dict_faterrcnn.values())
            print_logs['SupEncoderLoss'] = SupEncoderLoss
            losses += SupEncoderLoss

            imgs = list(transpose(i, None) for i in imgs)
            imgs = [i[0].to(auxdevice) for i in imgs]
            target_fasterrcnn = [{k: v.to(auxdevice) for k, v in t.items()} for t in target_fasterrcnn]
            add_fasterrcnn.append((imgs, target_fasterrcnn))

        if epoch >= StartSemiEpoch:
            moasics = []
            moasic_targetFRCNN = []
            model.eval()
            try:
                weak_samples, targets_semi, raw_strong = next(data_loader_semi)
            except StopIteration:
                data_loader_semi = iter(dataloader_semi)
                weak_samples, targets_semi, raw_strong = next(data_loader_semi)
            weak_samples = weak_samples.to(device)
            model.ema.apply_shadow()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp):
                    weak_output, memory = model(weak_samples, retmemory=True)
                weak_supervise, raw_strong = semi.filter_weak_output(weak_output, raw_strong)
                img = deepcopy(raw_strong)
                img = list(transpose(i, None) for i in img)
                img = [i[0].to(auxdevice) for i in img]
                weak_supervise = [{k: v.to(auxdevice) for k, v in t.items()} for t in weak_supervise]
                memory = model.fasterrcnn_ori(img, oriret=True, ori=True)
                weak_supervise = semi.labeloffset(weak_supervise, offset=1)
                target_refine = semi.RefinePseudo(weak_supervise, model, memory, [(i.size[1], i.size[0]) for i in raw_strong])
                weak_supervise = semi.labeloffset(target_refine, offset=-1)
                weak_supervise = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in weak_supervise]
            model.ema.restore()

            if raw_strong:
                mixup_dino, moasic_dino, mixup_fasterrcnn, moasic_fasterrcnn = semi.strong_aug(
                    raw_strong,
                    weak_supervise,
                    model.mixpl)
                if moasic_dino is not None:
                    moasics.append(moasic_dino)
                    moasic_targetFRCNN.append(moasic_fasterrcnn)
                if mixup_dino is not None:
                    strong_samples, weak_supervise = utils.collate_fn(mixup_dino)
                    mixupnum += len(weak_supervise)
                    strong_samples = strong_samples.to(device)
                    model.train()
                    weak_supervise = [{k: v.to(device) for k, v in t.items()} for t in weak_supervise]
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        strong_output, memory = model(strong_samples, weak_supervise, retmemory=True)

                    target_fasterrcnn = []
                    images: List[Tuple[int, int]] = []
                    images_ori: List[Tuple[int, int]] = []
                    img_high = []
                    for i in range(4):
                        images.append((strong_samples.tensors.shape[-2], strong_samples.tensors.shape[-1]))
                        images_ori.append(mixup_fasterrcnn[i][0])
                        target_fasterrcnn.append(mixup_fasterrcnn[i][1])
                        img_high.append(mixup_fasterrcnn[i][2])
                    target_fasterrcnn = [{k: v.to(device) for k, v in t.items()} for t in target_fasterrcnn]
                    memory = model.FPN(memory)

                    loss_dict_semi = criterion(strong_output, weak_supervise)
                    semi_loss_plus = 2 * sum(
                        loss_dict_semi[k] * weight_dict[k] for k in loss_dict_semi.keys() if
                        k in weight_dict)
                    print_logs['UnSupMixupLoss'] = semi_loss_plus
                    losses += semi_loss_plus

                    loss_dict_faterrcnn = model.fasterrcnn(images, original_image_sizes=images_ori, features=memory, targets=target_fasterrcnn)
                    CrossEncoderLoss = 2 * coef * sum(loss for loss in loss_dict_faterrcnn.values())
                    losses += CrossEncoderLoss
                    print_logs['UnSupMixupLoss_Aux'] = CrossEncoderLoss

                    imgs = list(transpose(i, None) for i in img_high)
                    imgs = [i[0].to(auxdevice) for i in imgs]
                    target_fasterrcnn = [{k: v.to(auxdevice) for k, v in t.items()} for t in target_fasterrcnn]
                    add_fasterrcnn.append((imgs, target_fasterrcnn))
                else:
                    add_fasterrcnn.append(None)


            if len(moasics):
                moasic_samples, moasic_targets = utils.collate_fn(moasics)
                moasic_samples = moasic_samples.to(device)
                model.train()
                weak_supervise = [{k: v.to(device) for k, v in t.items()} for t in moasic_targets]
                with torch.cuda.amp.autocast(enabled=args.amp):
                    strong_output, memory = model(moasic_samples, weak_supervise, retmemory=True)

                target_fasterrcnn = []
                images: List[Tuple[int, int]] = []
                images_ori: List[Tuple[int, int]] = []
                img_high = []
                for i in range(len(moasic_targetFRCNN)):
                    images.append((moasic_samples.tensors.shape[-2], moasic_samples.tensors.shape[-1]))
                    images_ori.append(moasic_targetFRCNN[i][0])
                    target_fasterrcnn.append(moasic_targetFRCNN[i][1])
                    img_high.append(moasic_targetFRCNN[i][2])
                target_fasterrcnn = [{k: v.to(device) for k, v in t.items()} for t in target_fasterrcnn]
                memory = model.FPN(memory)

                loss_dict_semi = criterion(strong_output, weak_supervise)
                loss_moasic = sum(loss_dict_semi[k] * weight_dict[k] for k in loss_dict_semi.keys() if
                                  k in weight_dict)
                print_logs['UnSupMoasicLoss'] = loss_moasic
                losses += loss_moasic
                moasicnum += len(moasics)

                loss_dict_faterrcnn = model.fasterrcnn(images, original_image_sizes=images_ori, features=memory, targets=target_fasterrcnn)
                CrossEncoderLoss_moasic = coef * sum(loss for loss in loss_dict_faterrcnn.values())
                losses += CrossEncoderLoss_moasic
                print_logs['UnSupMoasicLoss_Aux'] = CrossEncoderLoss_moasic
                

                imgs = list(transpose(i, None) for i in img_high)
                imgs = [i[0].to(auxdevice) for i in imgs]
                target_fasterrcnn = [{k: v.to(auxdevice) for k, v in t.items()} for t in target_fasterrcnn]
                add_fasterrcnn.append((imgs, target_fasterrcnn))
            else:
                add_fasterrcnn.append(None)


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        if softreg:
            model.train()
            imgs, target_fasterrcnn = add_fasterrcnn[0]
            loss_dict_faterrcnn_ori = model.fasterrcnn_ori(imgs, targets=target_fasterrcnn, ori=True)
            SupLoss = sum(loss.to(device) for loss in loss_dict_faterrcnn_ori.values())
            print_logs['SupLoss'] = SupLoss
            l += SupLoss
            t += 1
            losses = SupLoss
            if add_fasterrcnn[1] is not None:
                imgs, target_fasterrcnn = add_fasterrcnn[1]
                loss_dict_faterrcnn_ori = model.fasterrcnn_ori(imgs, targets=target_fasterrcnn, ori=True)
                SupLoss_mixup = 2 * sum(loss.to(device) for loss in loss_dict_faterrcnn_ori.values())
                losses += SupLoss_mixup
            if add_fasterrcnn[2] is not None:
                imgs, target_fasterrcnn = add_fasterrcnn[2]
                loss_dict_faterrcnn_ori = model.fasterrcnn_ori(imgs, targets=target_fasterrcnn, ori=True)
                SupLoss_moasic = sum(loss.to(device) for loss in loss_dict_faterrcnn_ori.values())
                losses += SupLoss_moasic
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            
        model.ema.update()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break
    with open("SemiRecord.txt", "a") as f:
        f.write(
            f"Epoch {epoch} Info: PseudoMixupCount = {mixupnum}, PseudoMosaicCount = {moasicnum}, Loss_FasterR-CNN = {1.0 * l / t}, Add_Loss: {[(k, v.item()) for k, v in print_logs.items()]}" + '\n')
    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False,
             args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if args.save_results:
        import os.path as osp

        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None,
         logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                    "image_id": int(image_id),
                    "category_id": l,
                    "bbox": b,
                    "score": s,
                }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)

    return final_res
