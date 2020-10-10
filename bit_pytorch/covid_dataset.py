from torch.utils.data import  DataLoader, Dataset, random_split, ConcatDataset


class RXImagesFolder(Dataset):

    np2pil = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.ToPILImage()
    ])

    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda img: img * 2.0 - 1.0)        
    ])

    def __init__(self, folder, label):
        self.folder = folder
        self.label = label

        self.images = []
        for fmt in ['png', 'jpg', 'jpeg']:
            self.images += glob.glob(os.path.join(folder, f'*.{fmt}'))

        self.augments = None

    def __len__(self,):
        return len(self.images)

    def remove_outliers(self, image):
        p = [1, 99]
        p1, p2 = np.percentile(image[..., 0], p)
        
        image[image < p1] = p1
        image[image > p2] = p2
        
        return image
        
        
    def __getitem__(self, idx):
        im = cv2.imread(self.images[idx],1)
        im = cv2.resize(im, (512, 512))        

        # apply augmentations
        if self.augments is not None:
            im = self.np2pil(im)
            im = self.augments(im)

        # to tensor
        im = self.transforms(im)
        return im, self.label

 
plit_dataset(dataset, pct, seed=0):
    torch.random.manual_seed(seed)
    train_size = round(len(dataset) * pct)
    valid_size = len(dataset) - train_size
    return random_split(dataset, [train_size, valid_size])

def get_train_augmentations():
    return tv.transforms.Compose([
        tv.transforms.RandomAffine(degrees=(-5, 5), translate=(0.02, 0.02), scale=(0.98, 1.02)),
        tv.transforms.RandomHorizontalFlip(),
    ])

def get_valid_augmentations():
    return tv.transforms.Compose([
        tv.transforms.CenterCrop(512)
    ])

def add_augmentations_to_datasets(datasets, augmentations):
    for ds in datasets:
        ds.augments = augmentations

def prepare_data(datadir, pct, seed, train_tx, val_tx):
    
    COVID = 1
    NON = 0
    
    # data from the ideagov
    covid_ds = RXImagesFolder(os.path.join(datadir, 'COVID'), COVID)
    normal_ds = RXImagesFolder(os.path.join(datadir, 'NORMAIS'), NON)
    noncovid_ds = RXImagesFolder(os.path.join(datadir, 'NÃO COVID'), NON)

    # data from the article
    covid5k_covid_ds = RXImagesFolder(os.path.join(root, 'data_covid5k/train/covid'), COVID)
    covid5k_noncovid_ds = RXImagesFolder(os.path.join(root, 'data_covid5k/train/non'), NON)
    
    covid5k_test_1 = RXImagesFolder(os.path.join(root, 'data_covid5k/test/covid'), COVID)
    covid5k_test_2 = RXImagesFolder(os.path.join(root, 'data_covid5k/test/non/No_Finding'), NON)
    covid5k_test_3 = RXImagesFolder(os.path.join(root, 'data_covid5k/test/non/other_diseases'), NON)
    
    covid_ds_train, covid_ds_valid = split_dataset(covid_ds, pct, seed)
    normal_ds_train, normal_ds_valid = split_dataset(normal_ds, pct, seed)
    noncovid_ds_train, noncovid_ds_valid = split_dataset(noncovid_ds, pct, seed)

    # add train augmentations
    add_augmentations_to_datasets([
        covid_ds_train,
        normal_ds_train,
        noncovid_ds_train,
        covid5k_covid_ds,
        covid5k_noncovid_ds,
    ], train_tx)
    
    ds_train = ConcatDataset([
        covid_ds_train,
        normal_ds_train,
        noncovid_ds_train,
        covid5k_covid_ds,
        covid5k_noncovid_ds
    ])

    add_augmentations_to_datasets([
        covid_ds_valid,
        normal_ds_valid,
        noncovid_ds_valid,
    ], val_tx)

    ds_valid = ConcatDataset([
        covid_ds_valid,
        normal_ds_valid,
        noncovid_ds_valid,

    ])

    add_augmentations_to_datasets([
        covid5k_test_1,
        covid5k_test_2,
        covid5k_test_3
    ], val_tx)
    
    ds_test = ConcatDataset([
        covid5k_test_1,
        covid5k_test_2,
        covid5k_test_3
    ])

    return ds_train, ds_valid, ds_test
