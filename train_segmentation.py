import torch

from rfdetr import RFDETRBase


# Create model with segmentation enabled
model = RFDETRBase(
    num_classes=3,
    num_queries=100,
    hidden_dim=256,
    enable_segmentation=True,  # Enable the lightweight segmentation head
    resolution=112,  # Input resolution
    batch_size=2,
    epochs=2,
    lr=1e-4
)

# Reinitialize detection head for 3 classes
model.model.model.reinitialize_detection_head(num_classes=3)

print("Model created with lightweight segmentation head for edge deployment")
print(f"Model has segmentation enabled: {model.model.model.enable_segmentation}")

# Train the model with segmentation capabilities
model.train(
    dataset_dir='/home/sandeep/rf-detr/data/coco_format',
    epochs=2,
    batch_size=2,
    enable_segmentation=True,  # Ensure segmentation is enabled during training
    output_dir='segmentation_output'
)