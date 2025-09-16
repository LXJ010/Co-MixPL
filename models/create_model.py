import torchvision
from models import customfasterrcnn


def create_model(num_classes, coco_model=False):
    model = customfasterrcnn.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    if coco_model:
        return model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = customfasterrcnn.FastRCNNPredictor(in_features, num_classes)
    return model

    
