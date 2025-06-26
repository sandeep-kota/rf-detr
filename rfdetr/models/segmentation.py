import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict

class MaskHead(nn.Module):
    def __init__(self, hidden_dim: int, num_channels: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        
        # Upscaling layers for mask prediction
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(hidden_dim, num_channels, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: Tensor, features: List[Tensor]):
        # x has shape [batch_size, num_queries, hidden_dim]
        # features is a list of multi-scale feature maps from the backbone
        
        # Use the lowest resolution feature map
        feature_map = features[-1]
        batch_size, num_queries, hidden_dim = x.shape
        
        # Reshape x to [batch_size * num_queries, hidden_dim, 1, 1]
        x = x.reshape(batch_size * num_queries, hidden_dim, 1, 1)
        
        # Upsample to match feature map size
        mask_features = F.interpolate(x, size=feature_map.shape[-2:], mode='bilinear', align_corners=False)
        
        # Apply convolutions
        mask_features = self.relu(self.conv1(mask_features))
        mask_features = self.relu(self.conv2(mask_features))
        mask_features = self.relu(self.conv3(mask_features))
        mask_features = self.relu(self.conv4(mask_features))
        mask_features = self.conv5(mask_features)
        
        # Reshape to [batch_size, num_queries, num_channels, H, W]
        mask_features = mask_features.reshape(batch_size, num_queries, self.num_channels, feature_map.shape[-2], feature_map.shape[-1])
        
        return mask_features


class PostProcessSegm(nn.Module):
    def __init__(self, return_masks: bool = True):
        super().__init__()
        self.return_masks = return_masks
        
    def forward(self, outputs, target_sizes, return_masks=None):
        if return_masks is None:
            return_masks = self.return_masks
            
        out_logits = outputs['pred_logits']
        out_bbox = outputs['pred_boxes']
        
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        
        # Convert from center-xywh to xyxy format
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        # Convert from center-xywh to xyxy format
        x_c, y_c, w, h = boxes.unbind(-1)
        boxes = torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)
        
        results = []
        for s, l, b in zip(scores, labels, boxes):
            result = {
                'scores': s,
                'labels': l,
                'boxes': b,
            }
            
            if return_masks and 'pred_masks' in outputs:
                masks = outputs['pred_masks']
                # Get masks for the current batch
                masks = masks[0, topk_boxes[0]]
                
                # Interpolate masks to target size
                masks = F.interpolate(masks, size=(img_h[0], img_w[0]), mode='bilinear', align_corners=False)
                
                # Binarize masks
                masks = masks > 0.5
                result['masks'] = masks
            
            results.append(result)
            
        return results