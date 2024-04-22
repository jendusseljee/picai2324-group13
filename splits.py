from picai_baseline.splits.picai import train_splits, valid_splits

for fold, ds_config in train_splits.items():
    print(f"Training fold {fold} has cases: {ds_config['subject_list']}")

for fold, ds_config in valid_splits.items():
    print(f"Validation fold {fold} has cases: {ds_config['subject_list']}")