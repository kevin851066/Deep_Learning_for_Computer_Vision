import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# Mean & STD for ResNet
MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1).cuda()


def denormalize(imgs):
    return (imgs * STD) + MEAN


def normalize(imgs):
    return (imgs - MEAN) / STD

class HardTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=5, metric='L2'):
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance
        dist = get_pairwise_distance(inputs, inputs, metric=self.metric)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


def get_pairwise_distance(A, B, metric='L2'):
    """
    :param A: shape (N1, F)
    :param B: shape (N2, F)
    :param metric: the metric to measure distances with (['L2', 'cosine'])
    :return: shape (N1, N2)
    """
    if metric == 'L2':
        return torch.cdist(A, B, p=2)
    elif metric == 'cosine':
        dot = torch.matmul(A, B.t())
        A_norm = torch.norm(A, dim=1).view(-1, 1).expand(A.shape[0], B.shape[0])
        B_norm = torch.norm(B, dim=1).view(1, -1).expand(A.shape[0], B.shape[0])
        return 1 - dot / (A_norm * B_norm)
    else:
        raise NotImplementedError


def evaluate_rank1(query_feats, query_ids, gallery_feats, gallery_ids, metric='L2'):
    dist = get_pairwise_distance(query_feats, gallery_feats, metric=metric)
    top_idx = dist.argmin(dim=1)

    correct_cnt = (query_ids == gallery_ids[top_idx]).sum().item()
    total_cnt = query_ids.cpu().numpy().size
    return correct_cnt / total_cnt


def filter_csv(csv_path, mask_dir):
    df = pd.read_csv(csv_path)
    mask_dir_list = os.listdir(mask_dir)
    valid_rows = []
    for _, row in df.iterrows():
        img_id, img_name = row.iloc[:]
        mask_img_name = img_name[:-4] + '_mask.png'
        if mask_img_name in mask_dir_list:
            valid_rows.append((img_id, mask_img_name))
    new_df = pd.DataFrame(valid_rows)
    new_df.to_csv('reid_mask_list_train.csv', header=None, index=False)


def mean_iou_score(pred, labels, num_classes=9):
    """
    Compute mean IoU score over 9 classes
    """
    mean_iou = 0
    for i in range(num_classes):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / num_classes
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou


if __name__ == '__main__':
    filter_csv('../../data/atrw_anno_reid_train/reid_list_train.csv', '../../data/mask')
