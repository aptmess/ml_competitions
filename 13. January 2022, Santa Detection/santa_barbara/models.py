# import torch.nn as nn
# from efficientnet_pytorch import EfficientNet
#
#
# class EfficientNetB7(nn.Module):
#     def __init__(self):
#         super(EfficientNetB7, self).__init__()
#         self.model = EfficientNet.from_pretrained('efficientnet-b7', pretrained=True)
#
#         self.classifier_layer = nn.Sequential(
#             nn.Linear(2560, 512),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.2),
#             nn.Linear(512, 256),
#             nn.Linear(256, 3)
#         )
#
#     def forward(self, inputs):
#         x = self.model.extract_features(inputs)
#
#         # Pooling and final linear layer
#         x = self.model._avg_pooling(x)
#         x = x.flatten(start_dim=1)
#         x = self.model._dropout(x)
#         x = self.classifier_layer(x)
#         return x
