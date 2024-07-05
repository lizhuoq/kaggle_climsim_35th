from data_provider.data_loader import ClimSimSeq2Seq
# from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ClimSim1D': ClimSimSeq2Seq
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    # timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    drop_last = False
    batch_size = args.batch_size
    # freq = args.freq
    args.data_path = "train_set.parquet" if flag == 'train' else 'val_set.parquet'

    data_set = Data(
        args = args
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


def data_provider_2D(args, flag):
    Data = ClimSim2D
    drop_last = False
    shuffle_flag = False if flag == "test" else True
    data_set = Data(
        args, flag=flag
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader