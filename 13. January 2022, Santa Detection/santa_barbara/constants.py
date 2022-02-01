from torchvision import transforms as tfs

PREPROCESS = tfs.Compose(
    [
        tfs.ToTensor(),
        tfs.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)
