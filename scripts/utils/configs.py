dataset_root_paths = {
    "cifar10": "/net/scratch/zsarwar/Datasets/CIFAR10",
 }

dataset_configs = {
    "Imagenet": {'train': {"dogs": "df_imagenet_dogs_train.pkl", 
                           "bottom_50": "df_imagenet_bottom50_train.pkl",
                           "full": "df_imagenet_train.pkl"},
                 'val': {"dogs": "df_imagenet_dogs_val.pkl", 
                           "bottom_50": "df_imagenet_bottom50_val.pkl",
                           "full": "df_imagenet_val.pkl"}   
                },
    "Tsinghua": {"train": {"dogs": "df_tsinghua_train.pkl"},
                 'val': {"dogs": "df_tsinghua_val.pkl"}
                },
    "OpenImages": {"train": {"dogs": "df_oi_dogs.pkl" }},
    "food101": {"train": {"full": "df_food101_train.pkl"}, 
                'val': {"full": "df_food101_val.pkl"}},
    "uecfood256": {"train": {"full": "df_uec256_train.pkl" },
                    "val": {"full": "df_uec256_val.pkl"}},
    }