import os
import json
import numpy as np
import pandas as pd

def build_re_block(alpha_pts, cd_pts, cl_pts, re_value, n_interp=60, noise_copies=5, noise_std_cd=0.015, noise_std_cl=0.030):
    """
    Interpolates sparse literature data points across the alpha range,
    then augments it with symmetry mirroring and Gaussian noise.
    """
    alpha_pts = np.array(alpha_pts)
    cd_pts = np.array(cd_pts)
    cl_pts = np.array(cl_pts)
    
    # 1. Dense interpolation across alpha range
    alpha_dense = np.linspace(alpha_pts.min(), alpha_pts.max(), n_interp)
    cd_dense = np.interp(alpha_dense, alpha_pts, cd_pts)
    cl_dense = np.interp(alpha_dense, alpha_pts, cl_pts)
    
    rows = []
    for i in range(len(alpha_dense)):
        # Original point
        rows.append([alpha_dense[i], re_value, cd_dense[i], cl_dense[i]])
        
        # Symmetry flip: negative alpha -> negative Cl, same Cd
        # Teaches Magnus Effect symmetry
        if abs(alpha_dense[i]) > 0.05:  # avoid duplicate at zero
            rows.append([-alpha_dense[i], re_value, cd_dense[i], -cl_dense[i]])
            
        # Gaussian noise augmentation
        for _ in range(noise_copies):
            cd_noisy = max(0.05, cd_dense[i] + np.random.normal(0, noise_std_cd))
            cl_noisy = cl_dense[i] + np.random.normal(0, noise_std_cl)
            rows.append([alpha_dense[i], re_value, cd_noisy, cl_noisy])
            
            if abs(alpha_dense[i]) > 0.05:
                cd_noisy_sym = max(0.05, cd_dense[i] + np.random.normal(0, noise_std_cd))
                cl_noisy_sym = -cl_dense[i] + np.random.normal(0, noise_std_cl)
                rows.append([-alpha_dense[i], re_value, cd_noisy_sym, cl_noisy_sym])
                
    return pd.DataFrame(rows, columns=['alpha', 'Re', 'Cd', 'Cl'])

def prepare_dataset(raw_data_path, output_dir, n_interp=60, noise_copies=5, split_ratio=0.8, seed=42,
                    csv_name='pinn_dataset_multi_re.csv',
                    stats_name='normalization_stats.json',
                    train_name='train_data.json',
                    val_name='val_data.json'):
    """
    Loads raw literature data, runs augmentation, normalizes,
    splits, and saves final files.
    """
    np.random.seed(seed)
    
    with open(raw_data_path, 'r') as f:
        raw_data = json.load(f)
        
    all_blocks = []
    for re_str, data in raw_data.items():
        re_val = int(re_str)
        df_block = build_re_block(
            alpha_pts=data['alpha'],
            cd_pts=data['Cd'],
            cl_pts=data['Cl'],
            re_value=re_val,
            n_interp=n_interp,
            noise_copies=noise_copies
        )
        all_blocks.append(df_block)
        
    df = pd.concat(all_blocks, ignore_index=True)
    # Shuffle dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save the unnormalized dataset for plotting reference
    os.makedirs(output_dir, exist_ok=True)
    dataset_csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(dataset_csv_path, index=False)
    print(f"Augmented dataset saved to {dataset_csv_path} (Total rows: {len(df)})")
    
    X = df[['alpha', 'Re']].values.astype(np.float32)
    Y = df[['Cd', 'Cl']].values.astype(np.float32)
    
    # Compute normalization statistics
    X_mean = X.mean(axis=0).tolist()
    X_std = (X.std(axis=0) + 1e-8).tolist()
    
    Y_mean = Y.mean(axis=0).tolist()
    Y_std = (Y.std(axis=0) + 1e-8).tolist()
    
    stats = {
        'X_mean': X_mean,
        'X_std': X_std,
        'Y_mean': Y_mean,
        'Y_std': Y_std
    }
    
    stats_json_path = os.path.join(output_dir, stats_name)
    with open(stats_json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Normalization stats saved to {stats_json_path}")
    
    # Apply normalization
    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std
    
    # Split train/val
    split_idx = int(split_ratio * len(df))
    
    train_data = {
        'X_train': X_norm[:split_idx].tolist(),
        'Y_train': Y_norm[:split_idx].tolist()
    }
    
    val_data = {
        'X_val': X_norm[split_idx:].tolist(),
        'Y_val': Y_norm[split_idx:].tolist()
    }
    
    train_json_path = os.path.join(output_dir, train_name)
    val_json_path = os.path.join(output_dir, val_name)
    
    with open(train_json_path, 'w') as f:
        json.dump(train_data, f)
    with open(val_json_path, 'w') as f:
        json.dump(val_data, f)
        
    print(f"Split data saved: {train_json_path} and {val_json_path}")
    return stats

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare PINN Aerodynamics Dataset")
    parser.add_argument('--large', action='store_true', help="Generate large dataset with ~100k rows")
    parser.add_argument('--n_interp', type=int, default=None, help="Number of dense interpolation points")
    parser.add_argument('--noise_copies', type=int, default=None, help="Number of noisy copies per point")
    
    args = parser.parse_args()
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_dir, 'data', 'raw', 'literature_data.json')
    out_dir = os.path.join(project_dir, 'data')
    
    if args.large:
        # Default parameters to get ~100k rows: 909 interpolation points, 10 noise copies
        # 5 * (2 * 909 - 1) * (10 + 1) = 99,935 rows.
        n_interp = 909
        noise_copies = 10
        print(f"Generating large dataset (~100k rows) with n_interp={n_interp}, noise_copies={noise_copies}")
        prepare_dataset(
            raw_path, 
            out_dir, 
            n_interp=n_interp, 
            noise_copies=noise_copies,
            csv_name='pinn_dataset_100k.csv',
            stats_name='normalization_stats_large.json',
            train_name='train_data_large.json',
            val_name='val_data_large.json'
        )
    elif args.n_interp is not None or args.noise_copies is not None:
        n_interp = args.n_interp if args.n_interp is not None else 60
        noise_copies = args.noise_copies if args.noise_copies is not None else 5
        print(f"Generating custom dataset with n_interp={n_interp}, noise_copies={noise_copies}")
        prepare_dataset(
            raw_path, 
            out_dir, 
            n_interp=n_interp, 
            noise_copies=noise_copies
        )
    else:
        # Default behavior: generate original small dataset
        print("Generating default dataset (3,552 rows) with n_interp=60, noise_copies=5")
        prepare_dataset(raw_path, out_dir, n_interp=60, noise_copies=5)
