import h5py
import os
import torchvision as tv

from torch.utils.data import  Dataset


class RXImagesFolder(Dataset):

    np2pil = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.ToPILImage()
    ])

    def __init__(self, path, augments):

        h5_file = h5py.File(path, 'r')
        self.labels = h5_file['label'].value.astype('int64')
        self.images = h5_file['data'].value.astype('float32')
        self.classes = ['not covid', 'covid', 'rejection']
        self.augments = augments

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        if im.ndim == 2:
            im = im[:, :, None].repeat(3, -1)
        elif im.ndim == 3:
            im = im.transpose(1, 2, 0)
            assert im.shape[2] == 3

        # apply augmentations
        if self.augments is not None:
            im = self.np2pil(im)
            im = self.augments(im)

        return im, self.labels[idx]

 
def prepare_data(datadir, train_tx, valid_tx):
    
    ds_train = RXImagesFolder(os.path.join(datadir, 'xray_550_COVIDx_train_sample2.hdf5'), augments=train_tx)
    ds_valid = RXImagesFolder(os.path.join(datadir, 'xray_550_COVIDx_val_sample2.hdf5'), augments=valid_tx)
    ds_test = RXImagesFolder(os.path.join(datadir, 'xray_550_COVIDx_test_sample2.hdf5'), augments=valid_tx)
    
    return ds_train, ds_valid, ds_test
