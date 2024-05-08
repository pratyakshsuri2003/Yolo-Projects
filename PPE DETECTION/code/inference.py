from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from IPython.display import clear_output
from PIL import Image
from IPython.display import clear_output
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

# checkpoint_path = 'ckpt_best.pth'  # Relative path to the checkpoint file

best_model = models.get('yolo_nas_l',
                        num_classes=len(dataset_params['classes']),
                        # checkpoint_path='/content/nas_l_run/RUN_20231105_162241_827632/ckpt_best.pth')
                        checkpoint_path='/content/nas_l_run/RUN_20240401_013806_578452/ckpt_best.pth')

# predicting an image # testing
img_url = '/content/drive/MyDrive/major_project/test_00.jpg'
best_model.predict(img_url, conf=0.40).show()

