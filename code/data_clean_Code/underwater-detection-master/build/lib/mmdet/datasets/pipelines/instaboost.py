import numpy as np

from ..builder import PIPELINES
from mmdet.core.evaluation.class_names import *

@PIPELINES.register_module()
class InstaBoost(object):
    r"""Data augmentation method in `InstaBoost: Boosting Instance
    Segmentation Via Probability Map Guided Copy-Pasting
    <https://arxiv.org/abs/1908.07801>`_.

    Refer to https://github.com/GothicAi/Instaboost for implementation details.
    """

    def __init__(self,
                 action_candidate=('normal', 'horizontal', 'skip'),
                 action_prob=(1, 0, 0),
                 scale=(0.8, 1.2),
                 dx=15,
                 dy=15,
                 theta=(-1, 1),
                 color_prob=0.5,
                 hflag=False,
                 aug_ratio=0.5):
#         try:
#             import instaboostfast as instaboost
#         except ImportError:
#             raise ImportError(
#                 'Please run "pip install instaboostfast" '
#                 'to install instaboostfast first for instaboost augmentation.')
        import instaboost.InstaBoost as instaboost
        self.cfg = instaboost.InstaBoostConfig(action_candidate, action_prob,
                                               scale, dx, dy, theta,
                                               color_prob, hflag)
        self.aug_ratio = aug_ratio

    def _load_anns(self, results):
        labels = results['ann_info']['labels']
        masks = results['ann_info']['masks']
        bboxes = results['ann_info']['bboxes']
        n = len(labels)

        anns = []
        for i in range(n):
            label = labels[i]
            bbox = bboxes[i]
            mask = masks[i]
            x1, y1, x2, y2 = bbox
            # assert (x2 - x1) >= 1 and (y2 - y1) >= 1
            bbox = [x1, y1, x2 - x1, y2 - y1]
            anns.append({
                'category_id': label,
                'segmentation': mask,
                'bbox': bbox
            })

        return anns

    def _parse_anns(self, results, anns, img):
        gt_bboxes = []
        gt_labels = []
        gt_masks_ann = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            # TODO: more essential bug need to be fixed in instaboost
            if w <= 0 or h <= 0:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(ann['category_id'])
            gt_masks_ann.append(ann['segmentation'])
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        results['ann_info']['labels'] = gt_labels
        results['ann_info']['bboxes'] = gt_bboxes
        results['ann_info']['masks'] = gt_masks_ann
        results['img'] = img
        return results

    def __call__(self, results):
        img = results['img']
        orig_type = img.dtype
        anns = self._load_anns(results)
        if np.random.choice([0, 1], p=[1 - self.aug_ratio, self.aug_ratio]):
            try:
                import instaboost.InstaBoost as instaboost
            except ImportError:
                raise ImportError('Please run "pip install instaboostfast" '
                                  'to install instaboostfast first.')
            print(anns)
            print(img.shape)
            anns, img = instaboost.get_new_data(
                anns, img.astype(np.uint8), self.cfg, background=None)

        results = self._parse_anns(results, anns, img.astype(orig_type))
        
        ## 可视化确认结果无误
        filename = results['img_info']['file_name']
        import cv2
        from mmdet.core.visualization import imshow_det_bboxes
        img_out = results['img'].copy()
        imshow_det_bboxes(img_out, results['gt_bboxes'], results['gt_labels'], class_names=underwater_classes(), 
                          show=False, out_file='/home/featurize/work/underwater-object-detection/boost/' + filename)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(cfg={self.cfg}, aug_ratio={self.aug_ratio})'
        return repr_str