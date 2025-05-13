from torchvision import datasets, transforms

data_path = "~/data"

transform_color = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_gray = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

transform_color_224 = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 正则化
    ]
)


def fetch_dataset(name) -> datasets:
    train_dataset, test_dataset = None, None
    if name == "mnist":
        train_dataset = datasets.MNIST(
            root=data_path, train=True, download=True, transform=transform_gray
        )
        test_dataset = datasets.MNIST(
            root=data_path, train=False, download=True, transform=transform_gray
        )
    elif name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        test_dataset = datasets.CIFAR10(
            root=data_path,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    elif name == "food101":
        train_dataset = datasets.Food101(
            root=data_path, split="train", download=True, transform=transform_color_224
        )
        test_dataset = datasets.Food101(
            root=data_path, split="test", download=True, transform=transform_color_224
        )
    elif name == "imagenette":
        train_dataset = datasets.ImageFolder(
            root=f"{data_path}/imagenette2-320/train", transform=transform_color_224
        )
        test_dataset = datasets.ImageFolder(
            root=f"{data_path}/imagenette2-320/val", transform=transform_color_224
        )

    if train_dataset is None or test_dataset is None:
        raise ValueError(f"Unknown dataset: {name}")

    return train_dataset, test_dataset
