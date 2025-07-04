from tensorflow.keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
