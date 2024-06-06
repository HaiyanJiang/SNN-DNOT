import torch
from data.data_loaders import load_data_cifar, load_data_cifardvs, load_data_imagenet


def datapool(dataname, batch_size, num_workers, autoaug=False, cutout=False, distributed=False,
             cache_dataset=False, time_steps=None):
    """
    works for imagenet only: distributed=False, cache_dataset=False
    time_steps=None should be used only for load_data_cifardvs
    """

    # ### The data
    print("Loading the data")
    if dataname.lower() == 'cifar10' or dataname == 'cifar100':
        dataset_train, dataset_test, train_sampler, test_sampler = load_data_cifar(
            use_cifar10= (dataname.lower() == 'cifar10'), download=True,
            distributed=distributed, cutout=cutout, autoaug=autoaug)
    elif dataname == 'cifardvs':
        dataset_train, dataset_test, train_sampler, test_sampler = load_data_cifardvs(distributed=distributed)
    elif dataname =='imagenet':
        # dataset_train, dataset_test, train_sampler, test_sampler = load_data_imagenet_0(distributed=distributed)
        dataset_train, dataset_test, train_sampler, test_sampler = load_data_imagenet(
            cache_dataset, distributed)
    else:
        print("still not support this model")
        exit(0)

    print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')
    print(f'Shape of one data: {dataset_train.__getitem__(0)[0].shape}')
    print(dataset_train.__getitem__(0)[0].shape)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=True)

    print(f'===================== Dataset name: {dataname} batch size: {batch_size} ===============================')
    print(f'Length of train_loader: {len(train_loader)}; Length of test_loader: {len(test_loader)}!')
    print(f'Approximately, number of samples in train_loader: {len(train_loader)*batch_size}; number of samples in test_loader: {len(test_loader)*batch_size}!')
    return train_loader, test_loader, train_sampler