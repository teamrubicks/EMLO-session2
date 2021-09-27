from torchvision import transforms

train_transform = transforms.Compose(
    transforms=[
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.48216, 0.44653], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)
valid_transforms = transforms.Compose(
    transforms=[
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.48216, 0.44653], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)
