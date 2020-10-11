import h5py
import os
import torchvision as tv

from torch.utils.data import  Dataset


class RXImagesFolder(Dataset):

    np2pil = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.ToPILImage()
    ])

    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda img: img * 2.0 - 1.0)        
    ])

    def __init__(self, path, augments):

        h5_file = h5py.File(path, 'r')
        self.labels = h5_file['label'].value.astype('int64')
        self.images = h5_file['data'].value.astype('float32')
        self.classes = ['não covid', 'covid', 'rejeição']
        self.augments = augments

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx][None, :, :]

        # apply augmentations
        if self.augments is not None:
            im = self.np2pil(im)
            im = self.augments(im)

        # to tensor
        #im = self.transforms(im)
        return im, self.labels[idx]

 
def get_train_augmentations():
    return tv.transforms.Compose([
        tv.transforms.RandomAffine(degrees=(-5, 5), translate=(0.02, 0.02), scale=(0.98, 1.02)),
        tv.transforms.RandomHorizontalFlip(),
    ])


def get_valid_augmentations():
    return tv.transforms.Compose([
        tv.transforms.CenterCrop(512)
    ])


def prepare_data(datadir, train_tx, valid_tx):
    
    ds_train = RXImagesFolder(os.path.join(datadir, 'xray_550_COVIDx_train_sample2.hdf5'), augments=train_tx)
    ds_valid = RXImagesFolder(os.path.join(datadir, 'xray_550_COVIDx_val_sample2.hdf5'), augments=valid_tx)
    ds_test = RXImagesFolder(os.path.join(datadir, 'xray_550_COVIDx_test_sample2.hdf5'), augments=valid_tx)

    return ds_train, ds_valid, ds_test
