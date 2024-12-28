from torchvision import transforms
import utils
from PIL import Image

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, args):
        self.args = args
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image, image2):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image2))
        return crops


class DataAugmentationDINO2(object):
        def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, args):
            self.args = args
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.Resize((224, 224)),  # Adjust size based on model requirement
                # transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                # flip_and_color_jitter,
                # utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.Resize((224, 224)),  # Adjust size based on model requirement
                # transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                # flip_and_color_jitter,
                # utils.GaussianBlur(0.1),
                # utils.Solarization(0.2),
                normalize,
            ])

            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])

        def __call__(self, image, image2):
            crops = []
            crops.append(self.global_transfo1(image))
            crops.append(self.global_transfo2(image2))
            return crops
    