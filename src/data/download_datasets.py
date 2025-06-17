import os
from tdc.multi_pred import DTI
from tdc import BenchmarkGroup
import pandas as pd

# Set the base output directory
out_dir = 'datasets'
CACHE = "/shared/mirekl/tdc/data/"

# Create the output directory if it doesn't exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Function to save splits to CSV
def save_splits(dataset_name, train_df, test_df, sample_size=30000):
    # Create dataset directory if it doesn't exist
    dataset_dir = os.path.join(out_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Save train and test splits
    train_path = os.path.join(dataset_dir, 'train.csv')
    test_path = os.path.join(dataset_dir, 'test.csv')
    all_path = os.path.join(dataset_dir, 'all.csv') 
    sample_path = os.path.join(dataset_dir, 'sample.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Save combined all.csv
    all_df = pd.concat([train_df, test_df], ignore_index=True) 
    all_df.to_csv(all_path, index=False)
    
    # Create and save sample.csv with max sample_size rows
    if len(all_df) > sample_size:
        # Random sample without replacement
        sample_df = all_df.sample(n=sample_size, random_state=42)
    else:
        # If fewer than sample_size rows, use all data
        sample_df = all_df
    
    # Save the sample
    sample_df.to_csv(sample_path, index=False)
    
    print(f"Saved {dataset_name} dataset to {dataset_dir}")
    print(f"  Train: {train_df.shape[0]} rows")
    print(f"  Test: {test_df.shape[0]} rows")
    print(f"  All: {all_df.shape[0]} rows")
    print(f"  Sample: {sample_df.shape[0]} rows")


# Process BindingDB_Patent (from BenchmarkGroup)
# print("Processing BindingDB_Patent...")
# group = BenchmarkGroup(name='DTI_DG_Group', path=CACHE)
# benchmark = group.get('BindingDB_Patent') # is already log
# train_df, test_df = benchmark['train_val'], benchmark['test']
# save_splits('BindingDB_Patent', train_df, test_df)

# Process datasets from DTI
datasets = [
    'BindingDB_Kd', 
    'BindingDB_Ki',
    'BindingDB_IC50',
    'DAVIS',
    'KIBA' # is already log
]

convert = [
    True,
    True,
    True,
    True,
    False,
]

for dataset_name, should_convert in zip(datasets, convert):
    print(f"Processing {dataset_name}...")
    data = DTI(name=dataset_name, path=CACHE)
    data.harmonize_affinities(mode='max_affinity')
    
    if should_convert:
        data.convert_to_log(form="binding")
    
    splits = data.get_split()
    train_df, test_df = splits["train"], splits["test"]
    
    # Process each dataframe (both train and test)
    for df in [train_df, test_df]:
        y_min, y_max = df['Y'].min(), df['Y'].max()
        
        if 'KIBA' in dataset_name:
            # For KIBA: 0 is strongest (most active), max is weakest
            # Y_similarity: 1 for most active (min Y), 0 for least active
            df['Y_similarity'] = 1 - ((df['Y'] - y_min) / (y_max - y_min)) if y_max != y_min else 1
            
            # Y_distance: 0 for most active (min Y), 1 for least active
            df['Y_distance'] = 1 - df['Y_similarity']
        else:
            # For other datasets: higher Y values mean more active (after the -log conversion)
            # Y_similarity: 1 for most active (max Y), 0 for least active
            df['Y_similarity'] = (df['Y'] - y_min) / (y_max - y_min) if y_max != y_min else 0
            
            # Y_distance: 0 for most active (max Y), 1 for least active
            df['Y_distance'] = 1 - df['Y_similarity']
    
    save_splits(dataset_name, train_df, test_df)

print("\nAll datasets saved successfully to the 'datasets' directory")