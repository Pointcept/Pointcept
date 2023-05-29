import numpy as np
import torch
import torch.distributed as dist
import pointops
import os
from uuid import uuid4
from copy import deepcopy

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = \
                intersection_and_union_gpu(
                    pred, label, self.trainer.cfg.data.num_classes, self.trainer.cfg.data.ignore_index)
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info('Test: [{iter}/{max_iter}] '
                                     'Loss {loss:.4f} '.format(iter=i + 1,
                                                               max_iter=len(self.trainer.val_loader),
                                                               loss=loss.item()))
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
            m_iou, m_acc, all_acc))
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info('Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
                idx=i, name=self.trainer.cfg.data.names[i], iou=iou_class[i], accuracy=acc_class[i]))
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar('val/loss', loss_avg, current_epoch)
            self.trainer.writer.add_scalar('val/mIoU', m_iou, current_epoch)
            self.trainer.writer.add_scalar('val/mAcc', m_acc, current_epoch)
            self.trainer.writer.add_scalar('val/allAcc', all_acc, current_epoch)
        self.trainer.logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info('Best {}: {:.4f}'.format(
            "allAcc", self.trainer.best_metric_value))


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(1, input_dict["coord"].float(), input_dict["offset"].int(),
                                            input_dict["origin_coord"].float(), input_dict["origin_offset"].int())
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]
            intersection, union, target = \
                intersection_and_union_gpu(
                    pred, segment, self.trainer.cfg.data.num_classes, self.trainer.cfg.data.ignore_index)
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(iter=i + 1, max_iter=len(self.trainer.val_loader))
            if "origin_coord" in input_dict.keys():
                info = "Precise " + info
            self.trainer.logger.info(info + 'Loss {loss:.4f} '.format(iter=i + 1,
                                                                      max_iter=len(self.trainer.val_loader),
                                                                      loss=loss.item()))
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
            m_iou, m_acc, all_acc))
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info('Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
                idx=i, name=self.trainer.cfg.data.names[i], iou=iou_class[i], accuracy=acc_class[i]))
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar('val/loss', loss_avg, current_epoch)
            self.trainer.writer.add_scalar('val/mIoU', m_iou, current_epoch)
            self.trainer.writer.add_scalar('val/mAcc', m_acc, current_epoch)
            self.trainer.writer.add_scalar('val/allAcc', all_acc, current_epoch)
        self.trainer.logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info('Best {}: {:.4f}'.format(
            "mIoU", self.trainer.best_metric_value))


class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if (instance_id == -1):
            return
        self.instance_id     = int(instance_id)
        self.label_id    = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"]    = self.label_id
        dict["vert_count"]  = self.vert_count
        dict["med_dist"]    = self.med_dist
        dict["dist_conf"]   = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id     = int(data["instance_id"])
        self.label_id        = int(data["label_id"])
        self.vert_count      = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist    = float(data["med_dist"])
            self.dist_conf   = float(data["dist_conf"])

    def __str__(self):
        return "("+str(self.instance_id)+")"


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, num_classes, class_names, segment_ignore_index=(-1, )):
        self.segment_ignore_index = segment_ignore_index
        self.VALID_CLASS_IDS = list(set(range(num_classes)) - set(segment_ignore_index))
        self.CLASS_LABELS = [class_names[i] for i in self.VALID_CLASS_IDS]
        self.ID_TO_LABEL = {}
        self.LABEL_TO_ID = {}
        for i in range(len(self.VALID_CLASS_IDS)):
            self.LABEL_TO_ID[self.CLASS_LABELS[i]] = self.VALID_CLASS_IDS[i]
            self.ID_TO_LABEL[self.VALID_CLASS_IDS[i]] = self.CLASS_LABELS[i]
        self.overlaps = np.append(np.arange(0.5,0.95,0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float('inf')
        self.distance_confs = -float('inf')

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()
    
    def get_instances(self, ids, class_ids, class_labels, id2label):
        instances = {}
        for label in class_labels:
            instances[label] = []
        instance_ids = np.unique(ids)
        for id in instance_ids:
            if id == 0:
                continue
            inst = Instance(ids, id)
            if inst.label_id in class_ids:
                instances[id2label[inst.label_id]].append(inst.to_dict())
        return instances
    
    def make_pred_info(self, pred):
        pred_info = {}
        assert(pred['pred_classes'].shape[0] == pred['pred_scores'].shape[0] == pred['pred_masks'].shape[1])
        for i in range(len(pred['pred_classes'])):
            info = {}
            info["label_id"] = pred['pred_classes'][i]
            info["conf"] = pred['pred_scores'][i]
            info["mask"] = pred['pred_masks'][:,i]
            pred_info[uuid4()] = info # we later need to identify these objects
        return pred_info

    def assign_instances_for_scan(self, pred, gt_ids):
        pred_info = self.make_pred_info(pred)
        # get gt instances
        gt_instances = self.get_instances(gt_ids, self.VALID_CLASS_IDS, self.CLASS_LABELS, self.ID_TO_LABEL)
        # associate
        gt2pred = deepcopy(gt_instances)
        for label in gt2pred:
            for gt in gt2pred[label]:
                gt['matched_pred'] = []
        pred2gt = {}
        for label in self.CLASS_LABELS:
            pred2gt[label] = []
        num_pred_instances = 0

        # mask of void labels in the groundtruth
        bool_void = np.logical_not(np.in1d(gt_ids//1000, self.VALID_CLASS_IDS))

        # go thru all prediction masks
        for uuid in pred_info:
            label_id = int(pred_info[uuid]['label_id'])
            conf = pred_info[uuid]['conf']
            if not label_id in self.ID_TO_LABEL:
                continue
            label_name = self.ID_TO_LABEL[label_id]
            # read the mask
            pred_mask = pred_info[uuid]['mask']
            assert(len(pred_mask) == len(gt_ids))
            # convert to binary
            pred_mask = np.not_equal(pred_mask, 0)
            num = np.count_nonzero(pred_mask)
            if num < self.min_region_sizes:
                continue  # skip if empty

            pred_instance = {}
            pred_instance['uuid'] = uuid
            pred_instance['pred_id'] = num_pred_instances
            pred_instance['label_id'] = label_id
            pred_instance['vert_count'] = num
            pred_instance['confidence'] = conf
            pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

            # matched gt instances
            matched_gt = []
            # go thru all gt instances with matching label
            for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
                intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
                if intersection > 0:
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy['intersection']   = intersection
                    pred_copy['intersection'] = intersection
                    matched_gt.append(gt_copy)
                    gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
            pred_instance['matched_gt'] = matched_gt
            num_pred_instances += 1
            pred2gt[label_name].append(pred_instance)

        return gt2pred, pred2gt

    
    def evaluate_matches(self, matches):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap = np.zeros( (len(dist_threshes) , len(self.CLASS_LABELS) , len(overlaps)) , float )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for m in matches:
                    for p in matches[m]['pred']:
                        for label_name in self.CLASS_LABELS:
                            for p in matches[m]['pred'][label_name]:
                                if 'uuid' in p:
                                    pred_visited[p['uuid']] = False
                for li, label_name in enumerate(self.CLASS_LABELS):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for m in matches:
                        pred_instances = matches[m]['pred'][label_name]
                        gt_instances = matches[m]['gt'][label_name]
                        # filter groups in ground truth
                        gt_instances = [ gt for gt in gt_instances if gt['vert_count']>=min_region_size and gt['med_dist']<=distance_thresh and gt['dist_conf']>=distance_conf ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true  = np.ones ( len(gt_instances) )
                        cur_score = np.ones ( len(gt_instances) ) * (-float("inf"))
                        cur_match = np.zeros( len(gt_instances) , dtype=bool )
                        # collect matches
                        for (gti,gt) in enumerate(gt_instances):
                            found_match = False
                            num_pred = len(gt['matched_pred'])
                            for pred in gt['matched_pred']:
                                # greedy assignments
                                if pred_visited[pred['uuid']]:
                                    continue
                                overlap = float(pred['intersection']) / (gt['vert_count']+pred['vert_count']-pred['intersection'])
                                if overlap > overlap_th:
                                    confidence = pred['confidence']
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max( cur_score[gti] , confidence )
                                        min_score = min( cur_score[gti] , confidence )
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true  = np.append(cur_true,0)
                                        cur_score = np.append(cur_score,min_score)
                                        cur_match = np.append(cur_match,True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred['uuid']] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true  = cur_true [ cur_match==True ]
                        cur_score = cur_score[ cur_match==True ]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred['matched_gt']:
                                overlap = float(gt['intersection']) / (gt['vert_count']+pred['vert_count']-gt['intersection'])
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred['void_intersection']
                                for gt in pred['matched_gt']:
                                    # group?
                                    if gt['instance_id'] < 1000:
                                        num_ignore += gt['intersection']
                                    # small ground truth instances
                                    if gt['vert_count'] < min_region_size or gt['med_dist']>distance_thresh or gt['dist_conf']<distance_conf:
                                        num_ignore += gt['intersection']
                                proportion_ignore = float(num_ignore)/pred['vert_count']
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true,0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score,confidence)

                        # append to overall results
                        y_true  = np.append(y_true,cur_true)
                        y_score = np.append(y_score,cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort      = np.argsort(y_score)
                        y_score_sorted      = y_score[score_arg_sort]
                        y_true_sorted       = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds,unique_indices) = np.unique( y_score_sorted , return_index=True )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples      = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                        precision         = np.zeros(num_prec_recall)
                        recall            = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append( y_true_sorted_cumsum , 0 )
                        # deal with remaining
                        for idx_res,idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores-1]
                            tp = num_true_examples - cumsum
                            fp = num_examples      - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p  = float(tp)/(tp+fp)
                            r  = float(tp)/(tp+fn)
                            precision[idx_res] = p
                            recall   [idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.
                        recall   [-1] = 0.

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.)

                        stepWidths = np.convolve(recall_for_conv,[-0.5,0,0.5],'valid')
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float('nan')
                    ap[di,li,oi] = ap_current
        return ap
    

    def compute_averages(self, aps):
        d_inf = 0
        o50   = np.where(np.isclose(self.overlaps,0.5))
        o25   = np.where(np.isclose(self.overlaps,0.25))
        oAllBut25  = np.where(np.logical_not(np.isclose(self.overlaps,0.25)))
        avg_dict = {}
        #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
        avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,oAllBut25])
        avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
        avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
        avg_dict["classes"]  = {}
        for (li,label_name) in enumerate(self.CLASS_LABELS):
            avg_dict["classes"][label_name]             = {}
            #avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
            avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,oAllBut25])
            avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ d_inf,li,o50])
            avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ d_inf,li,o25])
        return avg_dict


    def eval(self):
        self.trainer.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        self.trainer.model.eval()
        matches = {}
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]
            
            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # hardcode for ignore label
            output_dict['pred_classes'][output_dict['pred_classes'] == -1] = 255
            segment[segment == -1] = 255
            instance[instance == -1] = 255
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(1, input_dict["coord"].float(), input_dict["offset"].int(),
                                            input_dict["origin_coord"].float(), input_dict["origin_offset"].int())
                idx = idx.cpu().flatten().long()
                output_dict['pred_masks'] = output_dict['pred_masks'][idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            scene_id = input_dict['scene_id'][0]
            gt_ids = (segment * 1000 + instance).cpu().numpy()
            gt2pred, pred2gt = self.assign_instances_for_scan(output_dict, gt_ids)
            matches[scene_id] = {}
            matches[scene_id]['gt'] = gt2pred
            matches[scene_id]['pred'] = pred2gt

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info('Test: [{iter}/{max_iter}] '
                                     'Loss {loss:.4f} '.format(iter=i + 1,
                                                               max_iter=len(self.trainer.val_loader),
                                                               loss=loss.item()))


        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        all_matches = comm.gather(matches, dst=0)
        if comm.get_rank() == 0:    
            for _matches in all_matches:
                matches.update(_matches)
        ap_scores = self.evaluate_matches(matches)
        avgs = self.compute_averages(ap_scores)
        all_ap  = avgs["all_ap"]
        all_ap_50  = avgs["all_ap_50%"]
        all_ap_25  = avgs["all_ap_25%"]
        self.trainer.logger.info('Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.'.format(
            all_ap, all_ap_50, all_ap_25))
        for (i, label_name) in enumerate(self.CLASS_LABELS):
            ap = avgs["classes"][label_name]["ap"]
            ap_50 = avgs["classes"][label_name]["ap50%"]
            ap_25 = avgs["classes"][label_name]["ap25%"]
            self.trainer.logger.info('Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}'.format(
                idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25))
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar('val/loss', loss_avg, current_epoch)
            self.trainer.writer.add_scalar('val/mAP', all_ap, current_epoch)
            self.trainer.writer.add_scalar('val/AP50', all_ap_50, current_epoch)
            self.trainer.writer.add_scalar('val/AP25', all_ap_25, current_epoch)
        self.trainer.logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
        self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver
