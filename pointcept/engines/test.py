"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""


import os
import time
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from ..utils.registry import Registry
from ..utils.logger import get_root_logger
from ..utils.misc import AverageMeter, intersection_and_union, intersection_and_union_gpu, make_dirs
from ..datasets.utils import collate_fn

TEST = Registry("test")


@TEST.register_module()
class SemSegTester(object):
    """SemSegTester
    for large outdoor point cloud
    """

    def __call__(self, cfg, test_loader, model):
        test_dataset = test_loader.dataset
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        model.eval()

        save_path = os.path.join(cfg.save_path, "result", "test_epoch{}".format(cfg.epochs))
        make_dirs(save_path)
        if "ScanNet" in cfg.dataset_type:
            sub_path = os.path.join(save_path, "submit")
            make_dirs(sub_path)
        pred_save, label_save = [], []
        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)
            pred_save_path = os.path.join(save_path, '{}_pred.npy'.format(data_name))
            label_save_path = os.path.join(save_path, '{}_label.npy'.format(data_name))
            if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
                logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(test_dataset), data_name))
                pred, label = np.load(pred_save_path), np.load(label_save_path)
            else:
                data_dict_list, label = test_dataset[idx]
                pred = torch.zeros((label.size, cfg.data.num_classes)).cuda()
                batch_num = int(np.ceil(len(data_dict_list) / cfg.batch_size_test))
                for i in range(batch_num):
                    s_i, e_i = i * cfg.batch_size_test, min((i + 1) * cfg.batch_size_test, len(data_dict_list))
                    input_dict = collate_fn(data_dict_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                    if cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs: be], :] += pred_part[bs: be]
                        bs = be
                    logger.info('Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}'.format(
                        idx + 1, len(test_dataset), data_name=data_name, batch_idx=i, batch_num=batch_num))
                pred = pred.max(1)[1].data.cpu().numpy()
                np.save(pred_save_path, pred)
                np.save(label_save_path, label)
            intersection, union, target = intersection_and_union(pred, label, cfg.data.num_classes,
                                                                 cfg.data.ignore_index)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info('Test: {} [{}/{}]-{} '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Accuracy {acc:.4f} ({m_acc:.4f}) '
                        'mIoU {iou:.4f} ({m_iou:.4f})'.format(data_name, idx + 1, len(test_dataset), label.size,
                                                              batch_time=batch_time, acc=acc, m_acc=m_acc,
                                                              iou=iou, m_iou=m_iou))
            pred_save.append(pred)
            label_save.append(label)
            if "ScanNet" in cfg.dataset_type:
                np.savetxt(os.path.join(save_path, "submit", '{}.txt'.format(data_name)),
                           test_dataset.class2id[pred].reshape([-1, 1]), fmt="%d")

        with open(os.path.join(save_path, "pred.pickle"), 'wb') as handle:
            pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_path, "label.pickle"), 'wb') as handle:
            pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.data.num_classes):
            logger.info('Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
                idx=i, name=cfg.data.names[i], iou=iou_class[i], accuracy=accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


@TEST.register_module()
class ClsTester(object):
    """ClsTester
    for classification dataset (modelnet40), containing multi scales voting
    """
    def __call__(self, cfg, test_loader, model):
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        model.eval()

        for i, input_dict in enumerate(test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(pred, label, cfg.data.num_classes,
                                                                     cfg.data.ignore_index)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info('Test: [{}/{}] '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Accuracy {accuracy:.4f} '.format(i + 1, len(test_loader),
                                                          batch_time=batch_time,
                                                          accuracy=accuracy))

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

        for i in range(cfg.data.num_classes):
            logger.info('Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
                idx=i, name=cfg.data.names[i], iou=iou_class[i], accuracy=accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


@TEST.register_module()
class PartSegTester(object):
    """PartSegTester
    """

    def __call__(self, cfg, test_loader, model):
        test_dataset = test_loader.dataset
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

        batch_time = AverageMeter()

        num_categories = len(test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        model.eval()

        save_path = os.path.join(cfg.save_path, "result", "test_epoch{}".format(cfg.epochs))
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * cfg.batch_size_test, min((i + 1) * cfg.batch_size_test, len(data_dict_list))
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info('Test: {} {}/{}, Batch: {batch_idx}/{batch_num}'.format(
                    data_name, idx + 1, len(test_dataset), batch_idx=i, batch_num=batch_num))
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = test_loader.dataset.categories[category_index]
            parts_idx = test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info('Test: {} [{}/{}] '
                        'Batch {batch_time.val:.3f} '
                        '({batch_time.avg:.3f}) '.format(
                data_name, idx + 1, len(test_loader), batch_time=batch_time))

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info('Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.'.format(ins_mIoU, cat_mIoU))
        for i in range(num_categories):
            logger.info('Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}'.format(
                idx=i, name=test_loader.dataset.categories[i],
                iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                iou_count=int(iou_count[i])))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
