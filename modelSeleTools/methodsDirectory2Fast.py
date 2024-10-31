from adapteacher.engine.trainer import ATeacherTrainer
from scipy.optimize import linear_sum_assignment
import copy
import torch
import torch.nn.functional as F

def FIS(cfg, model, dataloaders, max_repeat, bos=False):
    evaluator = ATeacherTrainer.build_evaluator(cfg, cfg.DATASETS.TEST[0])
    dataloader = dataloaders[1]
    torch.set_grad_enabled(False)

    res, preds_gallery = ATeacherTrainer.test(cfg, model, evaluators=[evaluator],
                                              data_loader=dataloader, perturb=False)

    reg_list = [x[0]["instances"].get("pred_boxes").tensor for x in preds_gallery]
    logits = [x[0]["instances"].get("scores_logits") for x in preds_gallery]    # Notion: Softmax before returning.

    results_bbox_per_img = []
    results_logits_per_img = []
    for reg in reg_list:
        bboxes_per_img = [reg[i].cpu().numpy() for i in range(len(reg))]
        results_bbox_per_img.append(bboxes_per_img)
    for logit in logits:
        logits_per_img = [logit[i].cpu().numpy() for i in range(len(logit))]
        results_logits_per_img.append(logits_per_img)

    scores_perModel = {}
    lambdas = [1]
    for key in lambdas:
        scores_perModel[key] = []
    for repeat_time in range(1):
        _lambda = lambdas[repeat_time]
        print("\nrepeat time: {}".format(repeat_time))
        model_copied = copy.deepcopy(model)
        _, preds_gallery_perturb = ATeacherTrainer.test(cfg, model_copied, evaluators=[evaluator], 
                                                        data_loader=dataloader, perturb=True)
        reg_list_perturbe = [x[0]["instances"].get("pred_boxes").tensor for x in preds_gallery_perturb]
        logits_perturbe = [x[0]["instances"].get("scores_logits") for x in preds_gallery_perturb]

        results_bbox_per_img_perturbe = []
        results_logits_per_img_perturbe = []
        for reg in reg_list_perturbe:
            bboxes_per_img = [reg[i].cpu().numpy() for i in range(len(reg))]
            results_bbox_per_img_perturbe.append(bboxes_per_img)
        for logit in logits_perturbe:
            logits_per_img = [logit[i].cpu().numpy() for i in range(len(logit))]
            results_logits_per_img_perturbe.append(logits_per_img)

        iou_loss = IoUCost()
        num_flag = 0
        least_cost_final = 0

        for img_idx in range(len(results_logits_per_img)):
            _bboxes = results_bbox_per_img[img_idx]
            _bboxes_perturbe = results_bbox_per_img_perturbe[img_idx]
            _logits = results_logits_per_img[img_idx]
            _logits_perturbe = results_logits_per_img_perturbe[img_idx]

            _bboxes = torch.Tensor(_bboxes)
            _bboxes_perturbe = torch.Tensor(_bboxes_perturbe)
            _logits = torch.Tensor(_logits)
            _logits_perturbe = torch.Tensor(_logits_perturbe)

            if len(_bboxes.shape) < 2 or len(_bboxes_perturbe.shape) < 2:
                continue

            sampleCnt, _ = _bboxes.shape
            sampleCntPerturb, _ = _bboxes_perturbe.shape
            max_match = min(sampleCnt, sampleCntPerturb)
            num_flag += 1

            iou_cost = iou_loss(_bboxes_perturbe, _bboxes)
            kl_div = computeKLDivergenceMatrix(_logits_perturbe, _logits)

            costMatrix = iou_cost + kl_div * _lambda
            matched_rowIdx, matched_colIdx = linear_sum_assignment(costMatrix)
            least_cost_final += costMatrix[matched_rowIdx, matched_colIdx].sum().numpy().tolist() / max_match
        least_cost_final /= num_flag

        scores_perModel[_lambda].append(round(least_cost_final, 4))
        print((scores_perModel))
        del model_copied

    ground_truth = res["bbox"]["AP50"]

    return {"score": scores_perModel,
            "ground_truth": round(ground_truth, 4),}

def PDR(cfg, model, dataloaders, max_repeat):
    USE_BACKBONE_FEATURE = True
    assert len(dataloaders) == 2
    torch.set_grad_enabled(False)
    model.eval()

    dataloader_source, dataloader_target = dataloaders
    proto_target, _ = getPrototypes(model, dataloader_target,
                                                         USE_BACKBONE_FEATURE, [cfg.SEMISUPNET.DIS_TYPE])
    proto_source, _ = getPrototypes(model, dataloader_source,
                                                         USE_BACKBONE_FEATURE, [cfg.SEMISUPNET.DIS_TYPE])
    
    dist = calCateProtoDistance(proto_source, proto_target)
    
    return {"score": round(dist, 4),}

def calCateProtoDistance(source: torch.Tensor, target: torch.Tensor):
    cateNum, featureDim = source.shape

    l2_dist_crossDomain = torch.cdist(source, target)
    l2_dist_sourceInDomain = torch.cdist(source, source)
    l2_dist_targetInDomain = torch.cdist(target, target)

    dist_crossDomain_sameCate = l2_dist_crossDomain.diag().sum() / cateNum
    dist_crossDomain_diffCate = (l2_dist_crossDomain.sum() - l2_dist_crossDomain.diag().sum()) / ((cateNum-1) * cateNum)
    dist_sourceInDomain_diffCate = (l2_dist_sourceInDomain.sum() - l2_dist_sourceInDomain.diag().sum()) / ((cateNum-1) * cateNum)
    dist_targetInDomain_diffCate = (l2_dist_targetInDomain.sum() - l2_dist_targetInDomain.diag().sum()) / ((cateNum-1) * cateNum)

    # print(dist_crossDomain_diffCate, dist_sourceInDomain_diffCate, dist_targetInDomain_diffCate, dist_crossDomain_sameCate)
    div = round(((dist_crossDomain_diffCate * dist_sourceInDomain_diffCate * dist_targetInDomain_diffCate / dist_crossDomain_sameCate)).cpu().item(), 4)

    return div

def getPrototypes(model, dataloader, useBackboneFeature=True, featureNames=["vgg4"]):
    prototypes = []
    weights = []
    backboneFeatureList = []

    for idx, inputs in enumerate(dataloader):
        print("\robtaining prototypes: {}".format(idx), end="") if idx % 10 == 0 else ...
        assert not model.training

        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)

        features = [features[f] for f in featureNames]
        backboneFeatureList.append(torch.mean(features[0], dim=(2,3)))

        box_features_backbone = model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])  # [1000, 512, 7, 7]
        box_features_mlp = model.roi_heads.box_head(box_features_backbone)  # [1000, 1024]

        predictions = model.roi_heads.box_predictor(box_features_mlp)
        category_predictions = F.softmax(predictions[0], dim=1)

        if useBackboneFeature:
            instance_features = F.adaptive_avg_pool2d(box_features_backbone, (1, 1)).flatten(start_dim=1)
        else:
            instance_features = box_features_mlp
        
        instance_features_expand = instance_features.unsqueeze(1)       # [1000, 1, featureDim]
        category_predictions_expand = category_predictions.unsqueeze(2) # [1000, classNum, 1]

        prototypes_per_image = instance_features_expand * category_predictions_expand   # [1000, classNum, featureDim]
        prototypes_sum_per_image = prototypes_per_image.sum(dim=0)      # [classNum, featureDim]
        weights_sum_per_image = category_predictions.sum(dim=0)         # [classNum]

        prototypes.append(prototypes_sum_per_image)
        weights.append(weights_sum_per_image)
    
    backboneFeatureCat = torch.cat(backboneFeatureList, dim=0)

    total_prototype = torch.stack(prototypes).sum(dim=0)
    total_weight = torch.stack(weights).sum(dim=0)

    normalized_prototype = total_prototype / total_weight.unsqueeze(1)
    return normalized_prototype, backboneFeatureCat

def computeKLDivergenceMatrix(logits1: torch.Tensor, logits2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    log_probs1 = logits1.log().unsqueeze(1)
    log_probs2 = logits2.log().unsqueeze(0)
    
    kl_matrix = torch.sum(logits1.unsqueeze(1) * (log_probs1 - log_probs2), dim=-1)

    return kl_matrix

def _fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)

def _bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = _fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = _fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = _fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

class IoUCost():
    def __init__(self, iou_mode='iou', weight=1.): 
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        overlaps = _bbox_overlaps(bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight