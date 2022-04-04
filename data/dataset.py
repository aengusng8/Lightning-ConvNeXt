import torch
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from torchvision import transforms
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, image_path, mode, args):
        self.image_path = f"{image_path}/test" if mode == "test" else f"{image_path}/train" 
        self.mode = mode
        self.args = args

        if mode == "train":
            self.df = df[df['is_valid'] == False]
        elif mode == "val":
            self.df = df[df['is_valid'] == True]
        else:
            self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        fname = row["fname"]
        label = torch.Tensor(eval(row["multi_one_hot_label"]))

        image = Image.open(f"{self.image_path}/{fname}")
        transforms = self.build_transform(self.mode)
        image = transforms(image)

        return image, label

    def build_transform(self, mode):
        args = self.args
        resize_im = args.input_size > 32
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        if mode == "train":
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)
            return transform

        t = []
        if resize_im:
            # warping (no cropping) when evaluated at 384 or larger
            if args.input_size >= 384:
                t.append(
                    transforms.Resize((args.input_size, args.input_size),
                                      interpolation=transforms.InterpolationMode.BICUBIC),
                )
                print(f"Warping {args.input_size} size input images...")
            else:
                if args.crop_pct is None:
                    args.crop_pct = 224 / 256
                size = int(args.input_size / args.crop_pct)
                t.append(
                    # to maintain same ratio w.r.t. 224 images
                    transforms.Resize(
                        size, interpolation=transforms.InterpolationMode.BICUBIC),
                )
                t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)
