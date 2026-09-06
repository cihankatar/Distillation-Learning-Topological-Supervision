from torch.utils.data import DataLoader
from data.Custom_Dataset import dataset
from glob import glob
from torchvision.transforms import v2 
import os
import re
import torch
import torchvision.transforms.functional as TF

def data_transform():
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # --- CROP ONLY (used before RW) ---
    crop_transform_global = v2.Compose([
        v2.RandomResizedCrop(256, scale=(0.4, 1.0), ratio=(3/4, 4/3), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
    ])

    crop_transform_local = v2.Compose([
        v2.RandomResizedCrop(128, scale=(0.1, 0.4), ratio=(3/4, 4/3), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
    ])

    # --- COLOR ONLY (used after pseudo) ---
    color_transform_global = v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.3, 0.3, 0.3, 0.0)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=17, sigma=(0.1, 2.0))], p=0.3),
        #v2.RandomSolarize(threshold=0.5, p=0.2),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    color_transform_local = v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.3, 0.3, 0.3, 0.0)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=17, sigma=(0.1, 2.0))], p=0.5),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    return DinoMultiCropTransform(
        crop_transform_global, color_transform_global,
        crop_transform_local, color_transform_local,
        n_global=2, n_local=4
    )
class DinoMultiCropTransform:
    def __init__(self, crop_transform_global, color_transform_global,
                 crop_transform_local, color_transform_local,
                 n_global=2, n_local=4):
        
        self.crop_global = crop_transform_global
        self.color_global = color_transform_global
        self.crop_local = crop_transform_local
        self.color_local = color_transform_local
        self.n_global = n_global
        self.n_local = n_local

        # v2.RandomErasing'i buradan kaldırdık çünkü koordinatları içeride manuel yöneteceğiz
        self.erasing_p = 0.0
        self.erasing_scale = (0.05, 0.1)
        self.erasing_ratio = (0.3, 3.3)
        
        print(f"--- DinoMultiCropTransform Initialized ---")
        print(f"n_global: {self.n_global}")
        print(f"n_local : {self.n_local}")
        print(f"----------------------------------------")

    def __call__(self, img, real_mask, pseudo_mask):
        student_crops, teacher_crops, real_mask_crops = [], [], []
        pseudo_masks = []
        
        # Global views: teacher and student use the same spatial crop, with
        # independently sampled colour augmentations.  The pseudo/real masks
        # are cropped with the exact same coordinates.
        for _ in range(self.n_global):
            i, j, h, w = v2.RandomResizedCrop.get_params(img, scale=(0.4, 1.0), ratio=(3.0/4.0, 4.0/3.0))
            
            crop_img = TF.resized_crop(img, i, j, h, w, size=(256, 256), antialias=True)
            crop_mask = TF.resized_crop(real_mask, i, j, h, w, size=(256, 256), interpolation=TF.InterpolationMode.NEAREST)
            crop_pseudo = TF.resized_crop(pseudo_mask, i, j, h, w, size=(256, 256), interpolation=TF.InterpolationMode.NEAREST)

            if torch.rand(1) < 0.5:
                crop_img = TF.hflip(crop_img)
                crop_mask = TF.hflip(crop_mask)
                crop_pseudo = TF.hflip(crop_pseudo)

            # Tensor Dönüşümü (Erase ve Color işlemleri için zorunludur)
            if not isinstance(crop_img, torch.Tensor):
                crop_img = TF.to_tensor(crop_img)
            if not isinstance(crop_mask, torch.Tensor):
                crop_mask = TF.to_tensor(crop_mask)
            if not isinstance(crop_pseudo, torch.Tensor):
                crop_pseudo = TF.to_tensor(crop_pseudo)

            # Sadece görüntüye renk transformu
            # Separate calls sample different colour perturbations while
            # preserving spatial correspondence between teacher and student.
            student_crops.append(self.color_global(crop_img))
            teacher_crops.append(self.color_global(crop_img))
            real_mask_crops.append(crop_mask)
            pseudo_masks.append(crop_pseudo)

        # 4. Local crops for student
        for _ in range(self.n_local):
            cropped = self.crop_local(img)
            transformed = self.color_local(cropped)
            student_crops.append(transformed)

        return img, real_mask_crops, student_crops, teacher_crops, pseudo_masks


def _canonical_key(path):
    key = os.path.splitext(os.path.basename(path))[0].lower()

    # Prefer the dataset sample identifier wherever it occurs in the name.
    # This accepts variants such as ``ISIC_0000123_segmentation``,
    # ``topological_ISIC_0000123`` and ``ISIC-0000123-alpha-020`` without
    # relying on their method-specific prefixes or suffixes.
    isic_match = re.search(r"isic[\s_-]*(\d+)", key)
    if isic_match:
        return f"sample_{int(isic_match.group(1))}"

    for suffix in ("_segmentation", "_mask", "_lesion"):
        if key.endswith(suffix):
            key = key[:-len(suffix)]

    # PH2 and index-based exports commonly differ only in their textual
    # prefix. A trailing numeric identifier gives them the same safe key.
    numeric_match = re.search(r"(\d+)$", key)
    if numeric_match:
        return f"sample_{int(numeric_match.group(1))}"
    return key


def _align_triplets(images, masks, pseudo_masks, data, operation):
    indexed = []
    for name, paths in (("images", images), ("masks", masks), ("pseudo masks", pseudo_masks)):
        mapping = {_canonical_key(path): path for path in paths}
        if len(mapping) != len(paths):
            raise RuntimeError(f"Duplicate canonical filename in {data}/{operation}/{name}")
        indexed.append(mapping)

    key_sets = [set(mapping) for mapping in indexed]
    if not key_sets[0] or not (key_sets[0] == key_sets[1] == key_sets[2]):
        missing_from_masks = sorted(key_sets[0] - key_sets[1])[:5]
        missing_from_pmasks = sorted(key_sets[0] - key_sets[2])[:5]
        masks_without_images = sorted(key_sets[1] - key_sets[0])[:5]
        pmasks_without_images = sorted(key_sets[2] - key_sets[0])[:5]
        raise RuntimeError(
            f"Expected filename-aligned image/mask/pmask files for {data}/{operation}; "
            f"found {len(images)}, {len(masks)}, and {len(pseudo_masks)} files. "
            "Example differences: "
            f"images missing masks={missing_from_masks}, "
            f"images missing pmasks={missing_from_pmasks}, "
            f"masks without images={masks_without_images}, "
            f"pmasks without images={pmasks_without_images}."
        )
    ordered_keys = sorted(key_sets[0])
    return tuple([[mapping[key] for key in ordered_keys] for mapping in indexed])
    
    
def loader(op,mode,sslmode,batch_size,num_workers,image_size,cutout_pr,cutout_box,shuffle,split_ratio,data):

    if data=='isic_2018_1':
        foldernamepath="isic_2018_1/"
        imageext="/*.jpg"
        maskext="/*.png"
        pmaskext="/*.png"
    elif data == 'kvasir_1':
        foldernamepath="kvasir_1/"
        imageext="/*.jpg"
        maskext="/*.jpg"
        pmaskext="/*.png"
    elif data == 'ham_1':
        foldernamepath="HAM10000_1/"
        imageext="/*.jpg"
        maskext="/*.png"
        pmaskext="/*.png"
    elif data == 'PH2Dataset':
        foldernamepath="PH2Dataset/"
        imageext="/*.jpeg"
        maskext="/*.jpeg" 
        pmaskext="/*.png" 
    elif data == 'isic_2016_1':
        foldernamepath="isic_2016_1/"
        imageext="/*.jpg"
        maskext="/*.png"
        pmaskext="/*.png"
    else:
        raise ValueError(f"Unsupported dataset: {data}")

    data_root = os.environ.get("ML_DATA_ROOT")
    if not data_root:
        raise EnvironmentError("ML_DATA_ROOT must point to the directory containing the datasets")
    dataset_root = os.path.join(data_root, foldernamepath)

    # TopoDistill consumes the final alpha=0.20 topological masks written by
    # ``Pseudo Mask Extraction/test_new.py`` into each split's ``pmasks``
    # directory. The directory can still be overridden for an ablation.
    pseudo_mask_subdir = os.environ.get(
        "TOPODISTILL_PMASK_SUBDIR",
        "pmasks",
    )

    if op =="train":
        train_im_path   = os.path.join(dataset_root, "train/images")
        train_mask_path = os.path.join(dataset_root, "train/masks")
        train_pmask_path = os.path.join(dataset_root, "train", pseudo_mask_subdir)
        
        train_im_path   = sorted(glob(train_im_path+imageext))
        train_mask_path = sorted(glob(train_mask_path+maskext))
        train_pmask_path = sorted(glob(train_pmask_path+pmaskext))

    elif op == "validation":
        test_im_path    = os.path.join(dataset_root, "val/images")
        test_mask_path  = os.path.join(dataset_root, "val/masks")
        test_pmask_path = os.path.join(dataset_root, "val", pseudo_mask_subdir)

        test_im_path    = sorted(glob(test_im_path+imageext))
        test_mask_path  = sorted(glob(test_mask_path+maskext))
        test_pmask_path = sorted(glob(test_pmask_path+pmaskext))

    else :
        test_im_path    = os.path.join(dataset_root, "test/images")
        test_mask_path  = os.path.join(dataset_root, "test/masks")
        test_pmask_path = os.path.join(dataset_root, "test", pseudo_mask_subdir)
        test_im_path    = sorted(glob(test_im_path+imageext))
        test_mask_path  = sorted(glob(test_mask_path+maskext))
        test_pmask_path = sorted(glob(test_pmask_path+pmaskext))

    transformations = data_transform()

    if op == "train":
        paths = (train_im_path, train_mask_path, train_pmask_path)
    else:
        paths = (test_im_path, test_mask_path, test_pmask_path)
    paths = _align_triplets(*paths, data, op)
    if op == "train":
        train_im_path, train_mask_path, train_pmask_path = paths
    else:
        test_im_path, test_mask_path, test_pmask_path = paths

    if op == "train":
        data_train = dataset(
            train_im_path,
            train_mask_path,
            train_pmask_path,
            cutout_pr,
            cutout_box,
            transformations,
            mode,
        )
    else:
        data_test = dataset(
            test_im_path,
            test_mask_path,
            test_pmask_path,
            cutout_pr,
            cutout_box,
            transformations,
            mode,
        )

    if op == "train":
        train_loader = DataLoader(
            dataset     = data_train,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            persistent_workers=num_workers > 0
            )
        return train_loader
    
    else :
        test_loader = DataLoader(
            dataset     = data_test,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
        )
    
        return test_loader


#loader()
