import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from adapteacher.modeling.GModule.utils.hungarian import hungarian
# from adapteacher.modeling.GModule.utils.ILP import ILP_solver

import numpy as np

INF = 100000000


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.

    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs.softmax(dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        # pt = torch.clamp(torch.sigmoid(_input), min=0.00001)
        # pt = torch.clamp(pt, max=0.99999)
        pt = _input
        alpha = self.alpha
        epsilon = 1e-6 
        pt = pt.clamp(min=epsilon, max=1-epsilon)

        # pos = torch.nonzero(target[:,1] > 0).squeeze(1).numel()
        # print(pos)

        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'pos':
            loss = torch.sum(loss) / (2 * pos)

        return loss


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super().__init__()

        self.loc_loss_type = loc_loss_type

    def forward(self, out, target, weight=None):
        pred_left, pred_top, pred_right, pred_bottom = out.unbind(1)
        target_left, target_top, target_right, target_bottom = target.unbind(1)

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1) / (area_union + 1)

        if self.loc_loss_type == 'iou':
            loss = -torch.log(ious)

        elif self.loc_loss_type == 'giou':
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(
                pred_right, target_right
            )
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
                pred_top, target_top
            )
            g_intersect = g_w_intersect * g_h_intersect + 1e-7
            gious = ious - (g_intersect - area_union) / g_intersect

            loss = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (loss * weight).sum() / weight.sum()

        else:
            return loss.mean()


def clip_sigmoid(input):
    out = torch.clamp(torch.sigmoid(input), min=1e-4, max=1 - 1e-4)

    return out


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        loss = (
            -(t == class_ids).float() * alpha * term1
            -((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )
        return loss.sum()


class FCOSLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.sizes = opt.sizes

        self.cls_loss = SigmoidFocalLoss(opt.gamma, opt.alpha)
        self.box_loss = IOULoss(opt.iou_loss_type)
        self.center_loss = nn.BCEWithLogitsLoss()

        self.center_sample = opt.center_sample
        self.strides = opt.fpn_strides
        self.radius = opt.pos_radius

    def prepare_target(self, points, targets):
        ex_size_of_interest = []

        for i, point_per_level in enumerate(points):
            size_of_interest_per_level = point_per_level.new_tensor(self.sizes[i])
            ex_size_of_interest.append(
                size_of_interest_per_level[None].expand(len(point_per_level), -1)
            )

        ex_size_of_interest = torch.cat(ex_size_of_interest, 0)
        n_point_per_level = [len(point_per_level) for point_per_level in points]
        point_all = torch.cat(points, dim=0)
        label, box_target = self.compute_target_for_location(
            point_all, targets, ex_size_of_interest, n_point_per_level
        )

        for i in range(len(label)):
            label[i] = torch.split(label[i], n_point_per_level, 0)
            box_target[i] = torch.split(box_target[i], n_point_per_level, 0)

        label_level_first = []
        box_target_level_first = []

        for level in range(len(points)):
            label_level_first.append(
                torch.cat([label_per_img[level] for label_per_img in label], 0)
            )
            box_target_level_first.append(
                torch.cat(
                    [box_target_per_img[level] for box_target_per_img in box_target], 0
                )
            )

        return label_level_first, box_target_level_first

    def get_sample_region(self, gt, strides, n_point_per_level, xs, ys, radius=1):
        n_gt = gt.shape[0]
        n_loc = len(xs)
        gt = gt[None].expand(n_loc, n_gt, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2

        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        begin = 0

        center_gt = gt.new_zeros(gt.shape)

        for level, n_p in enumerate(n_point_per_level):
            end = begin + n_p
            stride = strides[level] * radius

            x_min = center_x[begin:end] - stride
            y_min = center_y[begin:end] - stride
            x_max = center_x[begin:end] + stride
            y_max = center_y[begin:end] + stride

            center_gt[begin:end, :, 0] = torch.where(
                x_min > gt[begin:end, :, 0], x_min, gt[begin:end, :, 0]
            )
            center_gt[begin:end, :, 1] = torch.where(
                y_min > gt[begin:end, :, 1], y_min, gt[begin:end, :, 1]
            )
            center_gt[begin:end, :, 2] = torch.where(
                x_max > gt[begin:end, :, 2], gt[begin:end, :, 2], x_max
            )
            center_gt[begin:end, :, 3] = torch.where(
                y_max > gt[begin:end, :, 3], gt[begin:end, :, 3], y_max
            )

            begin = end

        left = xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - xs[:, None]
        top = ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - ys[:, None]

        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_boxes = center_bbox.min(-1)[0] > 0

        return is_in_boxes

    def compute_target_for_location(
        self, locations, targets, sizes_of_interest, n_point_per_level
    ):
        labels = []
        box_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        # all_bboxes, all_labels = targets
        
        for i in range(len(targets)):

            targets_per_img = targets[i]
            assert targets_per_img.mode == 'xyxy'
            bboxes = targets_per_img.box
            labels_per_img = targets_per_img.fields['labels']
            area = targets_per_img.area()
            # bboxes = all_bboxes.squeeze(0)
            # labels_per_img = all_labels.squeeze(0)
            # area = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            box_targets_per_img = torch.stack([l, t, r, b], 2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, n_point_per_level, xs, ys, radius=self.radius
                )

            else:
                is_in_boxes = box_targets_per_img.min(2)[0] > 0

            max_box_targets_per_img = box_targets_per_img.max(2)[0]

            is_cared_in_level = \
                (max_box_targets_per_img >= sizes_of_interest[:, [0]]) &\
                (max_box_targets_per_img <= sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_level == 0] = INF

            locations_to_min_area, locations_to_gt_id = locations_to_gt_area.min(1)

            box_targets_per_img = box_targets_per_img[
                range(len(locations)), locations_to_gt_id
            ]

            labels_per_img = labels_per_img[locations_to_gt_id]
            labels_per_img[locations_to_min_area == INF] = 0

            labels.append(labels_per_img)
            box_targets.append(box_targets_per_img)

        return labels, box_targets

    def compute_centerness_targets(self, box_targets):
        left_right = box_targets[:, [0, 2]]
        top_bottom = box_targets[:, [1, 3]]
        centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * \
                     (top_bottom.min(-1)[0] / top_bottom.max(-1)[0])

        return torch.sqrt(centerness)

    def forward(self, locations, cls_pred, box_pred, center_pred, targets):
        batch = cls_pred[0].shape[0]
        n_class = cls_pred[0].shape[1]

        labels, box_targets = self.prepare_target(locations, targets)

        cls_flat = []
        box_flat = []
        center_flat = []

        labels_flat = []
        box_targets_flat = []

        for i in range(len(labels)):
            cls_flat.append(cls_pred[i].permute(0, 2, 3, 1).reshape(-1, n_class))
            box_flat.append(box_pred[i].permute(0, 2, 3, 1).reshape(-1, 4))
            center_flat.append(center_pred[i].permute(0, 2, 3, 1).reshape(-1))

            labels_flat.append(labels[i].reshape(-1))
            box_targets_flat.append(box_targets[i].reshape(-1, 4))

        cls_flat = torch.cat(cls_flat, 0)
        box_flat = torch.cat(box_flat, 0)
        center_flat = torch.cat(center_flat, 0)

        labels_flat = torch.cat(labels_flat, 0)
        box_targets_flat = torch.cat(box_targets_flat, 0)

        pos_id = torch.nonzero(labels_flat > 0).squeeze(1)

        cls_loss = self.cls_loss(cls_flat, labels_flat.int()) / (pos_id.numel() + batch)

        box_flat = box_flat[pos_id]
        center_flat = center_flat[pos_id]

        box_targets_flat = box_targets_flat[pos_id]

        if pos_id.numel() > 0:
            center_targets = self.compute_centerness_targets(box_targets_flat)

            box_loss = self.box_loss(box_flat, box_targets_flat, center_targets)
            center_loss = self.center_loss(center_flat, center_targets)

        else:
            box_loss = box_flat.sum()
            center_loss = center_flat.sum()

        return cls_loss, box_loss, center_loss

class PermutationLoss(nn.Module):
    r"""
    Binary cross entropy loss between two permutations, also known as "permutation loss".
    Proposed by `"Wang et al. Learning Combinatorial Embedding Networks for Deep Graph Matching. ICCV 2019."
    <http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf>`_

    .. math::
        L_{perm} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} + (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(PermutationLoss, self).__init__()
        self.loss = BCEFocalLoss()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        # batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)
        if pred_dsmat.dim() == 3:
            pred_dsmat = pred_dsmat.squeeze()

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        # n_sum = torch.zeros_like(loss)
        # for b in range(batch_num):
        #     batch_slice = [b, slice(src_ns), slice(tgt_ns)] # slice(src_ns[b]), slice(tgt_ns[b])
        #     loss += F.binary_cross_entropy(
        #         pred_dsmat[batch_slice],
        #         gt_perm[batch_slice],
        #         reduction='sum')
        #     n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
        loss += self.loss(pred_dsmat,gt_perm)

        return loss


class CrossEntropyLoss(nn.Module):
    r"""
    Multi-class cross entropy loss between two permutations.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged cross-entropy loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            gt_index = torch.max(gt_perm[batch_slice], dim=-1).indices
            loss += F.nll_loss(
                torch.log(pred_dsmat[batch_slice]),
                gt_index,
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class PermutationLossHung(nn.Module):
    r"""
    Binary cross entropy loss between two permutations with Hungarian attention. The vanilla version without Hungarian
    attention is :class:`~src.loss_func.PermutationLoss`.

    .. math::
        L_{hung} &=-\sum_{i\in\mathcal{V}_1,j\in\mathcal{V}_2}\mathbf{Z}_{ij}\left(\mathbf{X}^\text{gt}_{ij}\log \mathbf{S}_{ij}+\left(1-\mathbf{X}^{\text{gt}}_{ij}\right)\log\left(1-\mathbf{S}_{ij}\right)\right) \\
        \mathbf{Z}&=\mathrm{OR}\left(\mathrm{Hungarian}(\mathbf{S}),\mathbf{X}^\text{gt}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    Hungarian attention highlights the entries where the model makes wrong decisions after the Hungarian step (which is
    the default discretization step during inference).

    Proposed by `"Yu et al. Learning deep graph matching with channel-independent embedding and Hungarian attention.
    ICLR 2020." <https://openreview.net/forum?id=rJgBd2NYPH>`_

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.

    A working example for Hungarian attention:

    .. image:: ../../images/hungarian_attention.png
    """
    def __init__(self):
        super(PermutationLossHung, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        dis_pred = hungarian(pred_dsmat, src_ns, tgt_ns)
        ali_perm = dis_pred + gt_perm
        ali_perm[ali_perm > 1.0] = 1.0 # Hung
        pred_dsmat = torch.mul(ali_perm, pred_dsmat)
        gt_perm = torch.mul(ali_perm, gt_perm)
        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_dsmat[b, :src_ns[b], :tgt_ns[b]],
                gt_perm[b, :src_ns[b], :tgt_ns[b]],
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
        return loss / n_sum


class OffsetLoss(nn.Module):
    r"""
    OffsetLoss Criterion computes a robust loss function based on image pixel offset.
    Proposed by `"Zanfir et al. Deep Learning of Graph Matching. CVPR 2018."
    <http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html>`_

    .. math::
        \mathbf{d}_i =& \sum_{j \in V_2} \left( \mathbf{S}_{i, j} P_{2j} \right)- P_{1i} \\
        L_{off} =& \sum_{i \in V_1} \sqrt{||\mathbf{d}_i - \mathbf{d}^{gt}_i||^2 + \epsilon}

    :math:`\mathbf{d}_i` is the displacement vector. See :class:`src.displacement_layer.Displacement` or more details

    :param epsilon: a small number for numerical stability
    :param norm: (optional) division taken to normalize the loss
    """
    def __init__(self, epsilon: float=1e-5, norm=None):
        super(OffsetLoss, self).__init__()
        self.epsilon = epsilon
        self.norm = norm

    def forward(self, d1: Tensor, d2: Tensor, mask: float=None) -> Tensor:
        """
        :param d1: predicted displacement matrix
        :param d2: ground truth displacement matrix
        :param mask: (optional) dummy node mask
        :return: computed offset loss
        """
        # Loss = Sum(Phi(d_i - d_i^gt))
        # Phi(x) = sqrt(x^T * x + epsilon)
        if mask is None:
            mask = torch.ones_like(mask)
        x = d1 - d2
        if self.norm is not None:
            x = x / self.norm

        xtx = torch.sum(x * x * mask, dim=-1)
        phi = torch.sqrt(xtx + self.epsilon)
        loss = torch.sum(phi) / d1.shape[0]

        return loss


class FocalLoss(nn.Module):
    r"""
    Focal loss between two permutations.

    .. math::
        L_{focal} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left((1-\mathbf{S}_{i,j})^{\gamma} \mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} +
        \mathbf{S}_{i,j}^{\gamma} (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs, :math:`\gamma` is the focal loss
    hyper parameter.

    :param gamma: :math:`\gamma` parameter for focal loss
    :param eps: a small parameter for numerical stability

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self, gamma=0., eps=1e-15):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged focal loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            x = pred_dsmat[b, :src_ns[b], :tgt_ns[b]]
            y = gt_perm[b, :src_ns[b], :tgt_ns[b]]
            loss += torch.sum(
                - (1 - x) ** self.gamma * y * torch.log(x + self.eps)
                - x ** self.gamma * (1 - y) * torch.log(1 - x + self.eps)
            )
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class InnerProductLoss(nn.Module):
    r"""
    Inner product loss for self-supervised problems.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged inner product loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss -= torch.sum(pred_dsmat[batch_slice] * gt_perm[batch_slice])
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class HammingLoss(torch.nn.Module):
    r"""
    Hamming loss between two permutations.

    .. math::
        L_{hamm} = \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}_{i,j} (1-\mathbf{X}^{gt}_{i,j}) +  (1-\mathbf{X}_{i,j}) \mathbf{X}^{gt}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    Firstly adopted by `"Rolinek et al. Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers.
    ECCV 2020." <https://arxiv.org/abs/2003.11657>`_

    .. note::
        Hamming loss is defined between two discrete matrices, and discretization will in general truncate gradient. A
        workaround may be using the `blackbox differentiation technique <https://arxiv.org/abs/1912.02175>`_.
    """
    def __init__(self):
        super(HammingLoss, self).__init__()

    def forward(self, pred_perm: Tensor, gt_perm: Tensor) -> Tensor:
        r"""
        :param pred_perm: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :return:
        """
        errors = pred_perm * (1.0 - gt_perm) + (1.0 - pred_perm) * gt_perm
        return errors.mean(dim=0).sum()


class ILP_attention_loss(nn.Module):
    r"""
    Integer Linear Programming (ILP) attention loss between two permutations.
    Proposed by `"Jiang et al. Graph-Context Attention Networks for Size-Varied Deep Graph Matching. CVPR 2022."
    <https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_Graph-Context_Attention_Networks_for_Size-Varied_Deep_Graph_Matching_CVPR_2022_paper.html>`_

    .. math::
        L_{perm} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left( \max\left(a_{i,j}, \mathbf{X}^{gt}_{i,j} \right) \log \mathbf{S}_{i,j} + (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs, and a_{i,j} is the ILP assignment result.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self, varied_size=True):
        super(ILP_attention_loss, self).__init__()
        self.varied_size = varied_size

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        if self.varied_size:
            dis_pred = ILP_solver(pred_dsmat+1, src_ns+1, tgt_ns+1, dummy=True)
        else:
            dis_pred = ILP_solver(pred_dsmat, src_ns, tgt_ns)
        ali_perm = dis_pred + gt_perm
        ali_perm[ali_perm >= 1.0] = 1
        pred_dsmat = torch.mul(ali_perm, pred_dsmat)
        gt_perm = torch.mul(ali_perm, gt_perm)

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss += F.binary_cross_entropy(
                pred_dsmat[batch_slice],
                gt_perm[batch_slice],
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class Distill_InfoNCE(torch.nn.Module):
    def __init__(self):
        super(Distill_InfoNCE, self).__init__()

    def forward(self, feature: Tensor, feature_m: Tensor, alpha: float, dynamic_temperature: Tensor,
                dynamic_temperature_m: Tensor) -> Tensor:
        graph1_feat = F.normalize(feature[0], dim=-1)
        graph2_feat = F.normalize(feature[1], dim=-1)

        # following the contrastive loss in "Learning Transferable Visual Models From Natural Language Supervision"
        sim_1to2 = dynamic_temperature.exp() * graph1_feat @ graph2_feat.T
        sim_2to1 = dynamic_temperature.exp() * graph2_feat @ graph1_feat.T

        # get momentum features
        with torch.no_grad():
            graph1_feat_m = F.normalize(feature_m[0], dim=-1)
            graph2_feat_m = F.normalize(feature_m[1], dim=-1)

            # momentum similiarity
            sim_1to2_m = dynamic_temperature_m.exp() * graph1_feat_m @ graph2_feat_m.T
            sim_2to1_m = dynamic_temperature_m.exp() * graph2_feat_m @ graph1_feat_m.T
            sim_1to2_m = F.softmax(sim_1to2_m, dim=1)
            sim_2to1_m = F.softmax(sim_2to1_m, dim=1)

            # online similiarity
            sim_targets = torch.zeros(sim_1to2_m.size()).to(graph1_feat.device)
            sim_targets.fill_diagonal_(1)

            # generate pseudo contrastive labels
            sim_1to2_targets = alpha * sim_1to2_m + (1 - alpha) * sim_targets
            sim_2to1_targets = alpha * sim_2to1_m + (1 - alpha) * sim_targets

        loss_i2t = -torch.sum(F.log_softmax(sim_1to2, dim=1) * sim_1to2_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_2to1, dim=1) * sim_2to1_targets, dim=1).mean()
        contrast_loss = (loss_i2t + loss_t2i) / 2
        return contrast_loss


class Distill_QuadraticContrast(torch.nn.Module):
    def __init__(self):
        super(Distill_QuadraticContrast, self).__init__()

    def normalize(self, x: Tensor):
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def forward(self, feature: Tensor, feature_m: Tensor, dynamic_temperature: Tensor,
                dynamic_temperature_m: Tensor) -> Tensor:
        graph1_feat = F.normalize(feature[0], dim=-1)
        graph2_feat = F.normalize(feature[1], dim=-1)
        batch_size = graph1_feat.shape[0]

        with torch.no_grad():
            graph1_feat_m = F.normalize(feature_m[0], dim=-1)
            graph2_feat_m = F.normalize(feature_m[1], dim=-1)
            sim_1to2_m = graph1_feat_m @ graph2_feat_m.T
            w = ((torch.diag(sim_1to2_m) / sim_1to2_m.sum(dim=1)) + (
                        torch.diag(sim_1to2_m) / sim_1to2_m.sum(dim=0))) / 2
            # normalize w
            w = self.normalize(w)
            w = torch.mm(w.unsqueeze(1), w.unsqueeze(0))
            w = self.normalize(w)

        # cross-graph similarity
        sim_1to2 = dynamic_temperature.exp() * graph1_feat @ graph2_feat.T
        sim_2to1 = dynamic_temperature.exp() * graph2_feat @ graph1_feat.T
        # within-graph similarity
        sim_1to1 = dynamic_temperature.exp() * graph1_feat @ graph1_feat.T
        sim_2to2 = dynamic_temperature.exp() * graph2_feat @ graph2_feat.T
        # within-graph consistency
        within_graph_loss = (w * (sim_1to1 - sim_2to2).square()).mean() * batch_size / \
                            (dynamic_temperature.exp() * dynamic_temperature.exp()) # using batch_size to scale the loss
        # cross-graph consistency
        cross_graph_loss = (w * (sim_1to2 - sim_2to1).square()).mean() * batch_size / \
                           (dynamic_temperature.exp() * dynamic_temperature.exp())
        graph_loss = within_graph_loss + cross_graph_loss

        return graph_loss

