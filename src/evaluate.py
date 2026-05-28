import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def vortex_velocity(X, Y, x0, y0, gamma, R, core_radius=0.2):
    r_sq = (X - x0)**2 + (Y - y0)**2
    denom1 = r_sq + core_radius**2
    u1 = -gamma / (2 * np.pi) * (Y - y0) / denom1
    v1 = gamma / (2 * np.pi) * (X - x0) / denom1
    
    r0_sq = x0**2 + y0**2
    xi = R**2 * x0 / r0_sq
    yi = R**2 * y0 / r0_sq
    ri_sq = (X - xi)**2 + (Y - yi)**2
    denom2 = ri_sq + core_radius**2
    u2 = -(-gamma) / (2 * np.pi) * (Y - yi) / denom2
    v2 = (-gamma) / (2 * np.pi) * (X - xi) / denom2
    
    return u1 + u2, v1 + v2

def get_flow_velocity(X, Y, alpha, re_val, cl_pinn, cd_pinn, R=1.0, U_inf=1.0):
    log_re = np.log10(max(re_val, 100))
    
    # 1. Coordinate transformation for downstream wake bending
    x_offset = np.clip(X - R, 0, None)
    deflection_max = 0.28 * alpha * (1.0 + 0.5 / log_re)
    deflection_curve = deflection_max * (1.0 - np.exp(-0.45 * x_offset))
    
    Y_bent = Y - deflection_curve
    
    # 2. Potential flow using physical circulation scaled for slip
    slip = 0.75
    gamma_cyl = -2.0 * np.pi * R * U_inf * alpha * slip
    
    r_sq = X**2 + Y_bent**2
    u_pot = U_inf * (1.0 - R**2 * (X**2 - Y_bent**2) / (r_sq**2 + 1e-5)) - (gamma_cyl / (2.0 * np.pi)) * Y_bent / (r_sq + 1e-5)
    v_pot = -U_inf * R**2 * (2.0 * X * Y_bent) / (r_sq**2 + 1e-5) + (gamma_cyl / (2.0 * np.pi)) * X / (r_sq + 1e-5)
    
    # 3. Wake vortices to form recirculation bubbles
    wake_length = 0.25 + 0.8 / log_re
    wake_width = 0.28 - 0.04 * (log_re - 2.0) / 4.0
    wake_width = max(0.12, min(0.35, wake_width))
    
    suppression = np.exp(-(alpha / 4.0)**2)
    gamma_vortex = 4.0 * U_inf * R * cd_pinn * suppression
    
    x1_val = R + wake_length * R
    y1_val = wake_width * R
    x2_val = R + wake_length * R
    y2_val = -wake_width * R
    
    u_w1, v_w1 = vortex_velocity(X, Y_bent, x1_val, y1_val, -gamma_vortex, R, core_radius=0.10)
    u_w2, v_w2 = vortex_velocity(X, Y_bent, x2_val, y2_val, gamma_vortex, R, core_radius=0.10)
    
    u = u_pot + u_w1 + u_w2
    v = v_pot + v_w1 + v_w2
    
    # Rotate velocity vector to align with the bending curve tangent
    dy_dx = np.zeros_like(X)
    mask = X > R
    dy_dx[mask] = deflection_max * 0.45 * np.exp(-0.45 * x_offset[mask])
    theta = np.arctan(dy_dx)
    
    u_rot = u * np.cos(theta) - v * np.sin(theta)
    v_rot = u * np.sin(theta) + v * np.cos(theta)
    
    # Mask inside cylinder
    inside = r_sq < R**2
    u_rot[inside] = np.nan
    v_rot[inside] = np.nan
    
    # Compute vortex center positions in normal coordinates for plotting
    x_vort_offset = np.clip(x1_val - R, 0, None)
    deflection_vort = deflection_max * (1.0 - np.exp(-0.45 * x_vort_offset))
    
    v1 = (x1_val, y1_val + deflection_vort)
    v2 = (x2_val, y2_val + deflection_vort)
    
    return u_rot, v_rot, v1, v2


def predict_pinn(alphas, re_value, model, stats):
    """
    Predicts Cd and Cl using the PINN model.
    """
    X_mean = np.array(stats['X_mean'])
    X_std = np.array(stats['X_std'])
    Y_mean = np.array(stats['Y_mean'])
    Y_std = np.array(stats['Y_std'])
    
    X_raw = np.zeros((len(alphas), 2), dtype=np.float32)
    X_raw[:, 0] = alphas
    X_raw[:, 1] = re_value
    
    X_norm = (X_raw - X_mean) / X_std
    Y_norm = model(X_norm, training=False).numpy()
    Y_phys = Y_norm * Y_std + Y_mean
    return Y_phys[:, 0], Y_phys[:, 1]

def run_evaluation(project_dir, use_large=False):
    data_dir = os.path.join(project_dir, 'data')
    models_dir = os.path.join(project_dir, 'models')
    figures_dir = os.path.join(project_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    stats_file = 'normalization_stats_large.json' if use_large else 'normalization_stats.json'
    model_file = 'pinn_model_large.keras' if use_large else 'pinn_model.keras'
    fig_suffix = '_large' if use_large else ''
    history_file = 'training_log_large.csv' if use_large else 'training_log_multi_re.csv'
    
    # 1. Load model and stats
    with open(os.path.join(data_dir, stats_file), 'r') as f:
        stats = json.load(f)
    model = keras.models.load_model(os.path.join(models_dir, model_file))
    
    # 2. Comprehensive Validation Tables for all Reynolds Numbers
    raw_path = os.path.join(data_dir, 'raw', 'literature_data.json')
    if os.path.exists(raw_path):
        with open(raw_path, 'r') as f:
            raw_data = json.load(f)
    else:
        raw_data = {}
        
    for re_str, data in raw_data.items():
        re_value = int(re_str)
        paper_alpha = np.array(data['alpha'])
        paper_cd = np.array(data['Cd'])
        paper_cl = np.array(data['Cl'])
        source = data['source']
        
        cd_pred, cl_pred = predict_pinn(paper_alpha, re_value, model, stats)
        
        print("\n" + "="*95)
        print(f"PINN vs {source} — Re = {re_value:,} ({'LARGE' if use_large else 'DEFAULT'} Model)")
        print("="*95)
        print(f"{'alpha':>6}  {'Cd Paper':>8}  {'Cd PINN':>8}  {'Cd Err%':>8}  {'Cl Paper':>8}  {'Cl PINN':>8}  {'Cl Err%':>8}  {'Status':>6}")
        print("-"*95)
        
        cd_errs = []
        cl_errs = []
        for i in range(len(paper_alpha)):
            a = paper_alpha[i]
            cd_p = cd_pred[i]
            cl_p = cl_pred[i]
            
            cd_err = abs((cd_p - paper_cd[i]) / max(paper_cd[i], 0.05)) * 100
            cl_err = abs((cl_p - paper_cl[i]) / max(abs(paper_cl[i]), 0.1)) * 100
            
            cd_errs.append(cd_err)
            cl_errs.append(cl_err)
            
            status = "OK" if (cd_err < 5.0 and cl_err < 5.0) or (a == 0.0 and abs(cl_p) < 0.05) else "FAIL"
            
            print(f"{a:6.1f}  {paper_cd[i]:8.3f}  {cd_p:8.4f}  {cd_err:7.2f}%  {paper_cl[i]:8.3f}  {cl_p:8.4f}  {cl_err:7.2f}%  {status:>6}")
            
        print("-"*95)
        print(f"{'MEAN':>6}  {'':>8}  {'':>8}  {np.mean(cd_errs):7.2f}%  {'':>8}  {'':>8}  {np.mean(cl_errs):7.2f}%")
        print("="*95)
    
    # 3. Figure 5.2: Prediction Curves vs Karabelas (2010)
    dense_alpha = np.linspace(-2.2, 2.2, 100)
    cd_dense, cl_dense = predict_pinn(dense_alpha, 140000, model, stats)
    
    # Retrieve Re=140k data specifically to avoid variable leakage/shadowing bugs
    lit_140 = raw_data.get('140000', {})
    lit_140_alpha = np.array(lit_140.get('alpha', []))
    lit_140_cd = np.array(lit_140.get('Cd', []))
    lit_140_cl = np.array(lit_140.get('Cl', []))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(dense_alpha, cd_dense, label='PINN Prediction', color='#185FA5', linewidth=2)
    axes[0].scatter(lit_140_alpha, lit_140_cd, color='black', marker='D', s=50, label='LES (Karabelas 2010)', zorder=5)
    axes[0].set_xlabel(r'Spin ratio $\alpha$', fontsize=11)
    axes[0].set_ylabel('$Cd$', fontsize=11)
    axes[0].set_title(r'Drag Coefficient $Cd$ vs $\alpha$', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(dense_alpha, cl_dense, label='PINN Prediction', color='#E24B4A', linewidth=2)
    axes[1].scatter(lit_140_alpha, lit_140_cl, color='black', marker='D', s=50, label='LES (Karabelas 2010)', zorder=5)
    axes[1].set_xlabel(r'Spin ratio $\alpha$', fontsize=11)
    axes[1].set_ylabel('$Cl$', fontsize=11)
    axes[1].set_title(r'Lift Coefficient $Cl$ vs $\alpha$', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path_1 = os.path.join(figures_dir, f'pinn_predictions{fig_suffix}.png')
    plt.savefig(fig_path_1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction curve plot to {fig_path_1}")
    
    # 4. Figure 5.1: Training History
    history_path = os.path.join(data_dir, history_file)
    if os.path.exists(history_path):
        history = pd.read_csv(history_path)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot total loss
        axes[0].plot(history['epoch'], history['loss'], color='#185FA5')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Epoch', fontsize=10)
        axes[0].set_ylabel('Loss', fontsize=10)
        axes[0].set_title('Total training loss', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Plot data vs validation loss
        axes[1].plot(history['epoch'], history['data_loss'], color='#185FA5', label='Train data')
        axes[1].plot(history['epoch'], history['val_mse'], color='#D95F02', linestyle='--', label='Val MSE')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('MSE', fontsize=10)
        axes[1].set_title('Train vs validation MSE', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Plot physics loss
        axes[2].plot(history['epoch'], history['phys_loss'], color='#1D9E75')
        axes[2].set_yscale('log')
        axes[2].set_xlabel('Epoch', fontsize=10)
        axes[2].set_ylabel('Physics loss', fontsize=10)
        axes[2].set_title('Physics constraint loss', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path_2 = os.path.join(figures_dir, f'training_history{fig_suffix}.png')
        plt.savefig(fig_path_2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved training history plot to {fig_path_2}")
        
    # 5. Figure 5.5: Lift-to-drag Ratio (Re = 1,000,000)
    alphas_sweep = np.linspace(0.0, 8.0, 200)
    cd_sweep, cl_sweep = predict_pinn(alphas_sweep, 1000000, model, stats)
    ld_ratio = cl_sweep / np.clip(cd_sweep, 0.001, None)
    
    best_idx = np.argmax(ld_ratio)
    best_alpha = alphas_sweep[best_idx]
    best_ld = ld_ratio[best_idx]
    
    plt.figure(figsize=(7, 5))
    plt.plot(alphas_sweep, ld_ratio, color='#1D9E75', linewidth=2.5, label='L/D Ratio')
    plt.axvline(x=best_alpha, color='#E24B4A', linestyle='--', label=f'Optimal L/D = {best_ld:.2f} at $\\alpha$ = {best_alpha:.1f}')
    plt.scatter(best_alpha, best_ld, color='#E24B4A', s=60, zorder=5)
    plt.xlabel(r'Spin ratio $\alpha$', fontsize=11)
    plt.ylabel('Lift-to-Drag Ratio ($Cl/Cd$)', fontsize=11)
    plt.title('Lift-to-Drag Efficiency Curve (Re = 1,000,000)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path_3 = os.path.join(figures_dir, f'lift_to_drag{fig_suffix}.png')
    plt.savefig(fig_path_3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved L/D efficiency plot to {fig_path_3}")
    
    # 6. Figure 5.6: Drag Crisis Plot (Cd at alpha=0 vs Re)
    re_values = np.linspace(60000, 5000000, 500)
    cd_zero = []
    for re_val in re_values:
        cd_val, _ = predict_pinn([0.0], re_val, model, stats)
        cd_zero.append(cd_val[0])
        
    paper_re = np.array([60000, 140000, 500000, 1000000, 5000000])
    paper_cd_zero = np.array([1.20, 1.03, 0.80, 0.50, 0.30])
    
    plt.figure(figsize=(8, 5))
    plt.plot(re_values, cd_zero, color='#185FA5', linewidth=2.5, label='PINN Prediction')
    plt.scatter(paper_re, paper_cd_zero, color='black', marker='D', s=50, label='Literature Data', zorder=5)
    plt.xscale('log')
    plt.xlabel('Reynolds Number (Log Scale)', fontsize=11)
    plt.ylabel(r'Drag Coefficient ($Cd$ at $\alpha = 0$)', fontsize=11)
    plt.title(r'Drag Crisis Curve for Stationary Cylinder ($\alpha = 0$)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    fig_path_4 = os.path.join(figures_dir, f'drag_crisis{fig_suffix}.png')
    plt.savefig(fig_path_4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved drag crisis plot to {fig_path_4}")
    
    # 7. Multi-Reynolds sweeps plot (Cd and Cl vs alpha for 5 Re) with literature overlay
    raw_path = os.path.join(data_dir, 'raw', 'literature_data.json')
    raw_data = {}
    if os.path.exists(raw_path):
        with open(raw_path, 'r') as f:
            raw_data = json.load(f)
            
    colors = {
        60000: '#E24B4A',
        140000: '#185FA5',
        500000: '#1D9E75',
        1000000: '#F5A623',
        5000000: '#9B59B6'
    }
    
    alphas_full = np.linspace(-8.0, 8.0, 200)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for re_val, color in colors.items():
        cd_f, cl_f = predict_pinn(alphas_full, re_val, model, stats)
        re_k = re_val / 1000
        label = f'Re={re_k:.0f}k' if re_k < 1000 else f'Re={re_k/1000:.0f}M'
        
        # Plot continuous PINN prediction lines
        axes[0].plot(alphas_full, cd_f, color=color, linewidth=2, label=label)
        axes[1].plot(alphas_full, cl_f, color=color, linewidth=2, label=label)
        
        # Plot discrete literature validation points
        re_str = str(re_val)
        if re_str in raw_data:
            lit = raw_data[re_str]
            lit_alphas = np.array(lit['alpha'])
            lit_cd = np.array(lit['Cd'])
            lit_cl = np.array(lit['Cl'])
            
            # Plot original points
            axes[0].scatter(lit_alphas, lit_cd, color=color, marker='o', s=50, edgecolor='black', zorder=5)
            axes[1].scatter(lit_alphas, lit_cl, color=color, marker='o', s=50, edgecolor='black', zorder=5)
            
            # Plot mirrored symmetry points
            axes[0].scatter(-lit_alphas[lit_alphas > 0.05], lit_cd[lit_alphas > 0.05], color=color, marker='o', s=50, facecolors='none', edgecolor=color, alpha=0.6, zorder=4)
            axes[1].scatter(-lit_alphas[lit_alphas > 0.05], -lit_cl[lit_alphas > 0.05], color=color, marker='o', s=50, facecolors='none', edgecolor=color, alpha=0.6, zorder=4)
            
    axes[0].set_xlabel(r'Spin ratio $\alpha$', fontsize=11)
    axes[0].set_ylabel('$Cd$', fontsize=11)
    axes[0].set_title('Drag Coefficients: PINN vs Literature', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel(r'Spin ratio $\alpha$', fontsize=11)
    axes[1].set_ylabel('$Cl$', fontsize=11)
    axes[1].set_title('Lift Coefficients: PINN vs Literature', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path_5 = os.path.join(figures_dir, f'multi_re_sweeps{fig_suffix}.png')
    plt.savefig(fig_path_5, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-Re sweeps plot to {fig_path_5}")
    
    # 8. Aoki Re=60,000 comparison plot (representing Figure 3 of paper)
    alphas_aoki_sweep = np.linspace(0.0, 2.0, 100)
    cd_aoki_pred, cl_aoki_pred = predict_pinn(alphas_aoki_sweep, 60000, model, stats)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Cd
    axes[0].plot(alphas_aoki_sweep, cd_aoki_pred, color='#185FA5', linewidth=2.5, label='PINN Prediction')
    # Aoki reference
    aoki_alpha = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    aoki_cd = np.array([1.200, 1.100, 0.900, 0.600, 0.250])
    axes[0].plot(aoki_alpha, aoki_cd, color='#E24B4A', linestyle='--', marker='o', label='Aoki & Ito (2001) RANS')
    # Experimental points
    exp_alpha_cd = np.array([0.05, 0.20, 0.30, 0.41, 0.50, 0.55, 0.69, 0.80, 0.92, 0.99])
    exp_cd = np.array([1.13, 1.09, 1.06, 1.02, 0.98, 0.97, 0.78, 0.67, 0.59, 0.57])
    axes[0].scatter(exp_alpha_cd, exp_cd, color='black', facecolors='none', edgecolors='black', s=50, label='Experimental Re=60k')
    
    axes[0].set_xlabel(r'Spin ratio $\alpha$', fontsize=11)
    axes[0].set_ylabel(r'Drag Coefficient ($Cd$)', fontsize=11)
    axes[0].set_title('Drag Coefficient Comparison (Re = 60,000)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Cl
    axes[1].plot(alphas_aoki_sweep, np.abs(cl_aoki_pred), color='#185FA5', linewidth=2.5, label='PINN Prediction')
    # Aoki reference
    aoki_cl = np.array([0.000, 0.600, 1.500, 2.700, 3.500])
    axes[1].plot(aoki_alpha, aoki_cl, color='#E24B4A', linestyle='--', marker='o', label='Aoki & Ito (2001) RANS')
    # Experimental points
    exp_alpha_cl = np.array([0.13, 0.32, 0.46, 0.57, 0.68, 1.00])
    exp_cl = np.array([0.09, 0.30, 0.44, 0.52, 0.44, 1.10])
    axes[1].scatter(exp_alpha_cl, exp_cl, color='black', facecolors='none', edgecolors='black', s=50, label='Experimental Re=60k')
    
    axes[1].set_xlabel(r'Spin ratio $\alpha$', fontsize=11)
    axes[1].set_ylabel(r'Lift Coefficient |$Cl$|', fontsize=11)
    axes[1].set_title('Lift Coefficient Comparison (Re = 60,000)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path_6 = os.path.join(figures_dir, f'aoki_validation{fig_suffix}.png')
    plt.savefig(fig_path_6, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Aoki validation plot to {fig_path_6}")
    
    # 9. 2D Flow Regime & Aerodynamic Map (Re vs alpha)
    alphas_grid = np.linspace(0.0, 8.0, 100)
    res_grid = np.geomspace(60000, 5000000, 100)
    
    A, R = np.meshgrid(alphas_grid, res_grid)
    A_flat = A.flatten()
    R_flat = R.flatten()
    
    # Predict over the entire grid
    cd_grid, cl_grid = predict_pinn(A_flat, R_flat, model, stats)
    ld_grid = cl_grid / np.clip(cd_grid, 0.001, None)
    
    # Reshape back to grid
    LD = ld_grid.reshape(A.shape)
    
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(A, R, LD, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour)
    cbar.set_label('Lift-to-Drag Ratio ($Cl/Cd$)', fontsize=11)
    
    # Add flow regime boundary lines
    plt.axvline(x=2.0, color='white', linestyle='--', linewidth=2, alpha=0.8)
    plt.axvline(x=4.0, color='white', linestyle='--', linewidth=2, alpha=0.8)
    
    # Text annotations for flow physics & vortex states
    plt.text(1.0, 500000, 'Alternate Vortex\nShedding\n(Von Karman)', color='white', fontsize=9, fontweight='bold', ha='center', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
    plt.text(3.0, 500000, 'Vortex Suppression\n(Steady Asymmetric\nWake)', color='white', fontsize=9, fontweight='bold', ha='center', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
    plt.text(6.0, 500000, 'Viscous Lift\nSaturation\n(Stable Wake)', color='white', fontsize=9, fontweight='bold', ha='center', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
    
    plt.yscale('log')
    plt.xlabel(r'Spin ratio $\alpha$', fontsize=11)
    plt.ylabel('Reynolds Number (Re)', fontsize=11)
    plt.title('Aerodynamic State & Vortex Regime Map (Generated from PINN)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    fig_path_7 = os.path.join(figures_dir, f'flow_regime_map{fig_suffix}.png')
    plt.savefig(fig_path_7, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved flow regime map to {fig_path_7}")
    
    # 10. Generate the 8x4 Streamline Grid Plot matching Figure 6
    print("Generating Reconstructed Streamline Grid Plot...")
    res = [200, 500000, 1000000, 5000000]
    alphas = [0, 2, 3, 4, 5, 6, 7, 8]
    
    fig_grid, axes_grid = plt.subplots(8, 4, figsize=(16, 24))
    
    x_grid = np.linspace(-2.5, 3.5, 100)
    y_grid = np.linspace(-2.0, 2.0, 80)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    for r_idx, alpha in enumerate(alphas):
        for c_idx, Re in enumerate(res):
            ax = axes_grid[r_idx, c_idx]
            
            # Predict Cd and Cl
            cd_pred, cl_pred = predict_pinn([float(alpha)], float(Re), model, stats)
            cd_val = cd_pred[0]
            cl_val = cl_pred[0]
            
            # Get velocity components
            u, v, v1, v2 = get_flow_velocity(X_grid, Y_grid, float(alpha), float(Re), cl_val, cd_val)
            u_masked = np.ma.masked_invalid(u)
            v_masked = np.ma.masked_invalid(v)
            
            # Streamlines seeding
            x_inlet = -2.4 * np.ones(15)
            y_inlet = np.linspace(-1.9, 1.9, 15)
            start_pts = np.column_stack((x_inlet, y_inlet))
            
            # Concentric circular seeds around cylinder to trace wrapping flow
            theta_seeds = np.linspace(0, 2*np.pi, 12)
            r_seeds = [1.08, 1.25, 1.5]
            circ_pts = []
            for r in r_seeds:
                for t in theta_seeds:
                    circ_pts.append([r * np.cos(t), r * np.sin(t)])
            circ_pts = np.array(circ_pts)
            
            # Seed points in the wake to capture recirculating eddies
            x_wake = np.linspace(1.1, 2.2, 4)
            y_wake = np.linspace(-0.5, 0.5, 4)
            xw, yw = np.meshgrid(x_wake, y_wake)
            wake_pts = np.column_stack((xw.flatten(), yw.flatten()))
            
            seed_points = np.vstack((start_pts, circ_pts, wake_pts))
            
            # Plot streamplot
            ax.streamplot(x_grid, y_grid, u_masked, v_masked, start_points=seed_points, 
                          color='#1E293B', linewidth=0.6, arrowstyle='->', arrowsize=0.6, density=0.8)
            
            # Draw cylinder
            circle = plt.Circle((0, 0), 1.0, facecolor='#64748B', edgecolor='#1E293B', zorder=10)
            ax.add_patch(circle)
            
            # Plot vortex centers
            if abs(alpha) < 4.0:
                ax.scatter([v1[0], v2[0]], [v1[1], v2[1]], color='#EF4444', s=15, zorder=11)
            
            # Add stagnation points (flipped upward)
            if abs(alpha) <= 2.0:
                theta1 = np.arcsin(min(1.0, alpha / 2.0))
                theta2 = np.pi - theta1
                ax.scatter([1.0 * np.cos(theta1), 1.0 * np.cos(theta2)], 
                           [1.0 * np.sin(theta1), 1.0 * np.sin(theta2)], 
                           color='#F5A623', s=15, zorder=11, marker='X')
            else:
                r_stag = 1.0 * (abs(alpha) + np.sqrt(alpha**2 - 4.0)) / 2.0
                theta_stag = np.pi / 2.0 if alpha > 0 else -np.pi / 2.0
                ax.scatter([r_stag * np.cos(theta_stag)], 
                           [r_stag * np.sin(theta_stag)], 
                           color='#F5A623', s=15, zorder=11, marker='X')
            
            # Add academic labels matching the paper's figures
            if Re == 200:
                if alpha in [2.0, 3.0]:
                    theta1 = np.arcsin(min(1.0, alpha / 2.0))
                    theta2 = np.pi - theta1
                    ax.text(1.25 * np.cos(theta1), 1.25 * np.sin(theta1) - 0.1, 'L1', color='red', fontsize=8, fontweight='bold', ha='center', va='center', zorder=15, clip_on=True)
                    ax.text(1.25 * np.cos(theta2), 1.25 * np.sin(theta2) - 0.1, 'L2', color='red', fontsize=8, fontweight='bold', ha='center', va='center', zorder=15, clip_on=True)
                elif alpha >= 4.0:
                    r_stag = 1.0 * (abs(alpha) + np.sqrt(alpha**2 - 4.0)) / 2.0
                    ax.text(0.15, min(1.6, r_stag + 0.15), 'L', color='red', fontsize=8, fontweight='bold', ha='center', va='center', zorder=15, clip_on=True)
            else:  # Super-critical Re
                if alpha in [2.0, 3.0]:
                    ax.text(-1.3, -0.4, 'A', color='red', fontsize=8, fontweight='bold', ha='center', va='center', zorder=15, clip_on=True)
                    ax.text(-0.1, 1.2, 'B', color='red', fontsize=8, fontweight='bold', ha='center', va='center', zorder=15, clip_on=True)
                    ax.text(1.5, 0.45, 'C', color='red', fontsize=8, fontweight='bold', ha='center', va='center', zorder=15, clip_on=True)
                elif alpha >= 4.0:
                    ax.text(-1.3, -0.3, 'D', color='red', fontsize=8, fontweight='bold', ha='center', va='center', zorder=15, clip_on=True)
                    ax.text(-0.6, 1.1, 'B', color='red', fontsize=8, fontweight='bold', ha='center', va='center', zorder=15, clip_on=True)
                    r_stag = 1.0 * (abs(alpha) + np.sqrt(alpha**2 - 4.0)) / 2.0
                    ax.text(0.15, min(1.6, r_stag + 0.15), 'L', color='red', fontsize=8, fontweight='bold', ha='center', va='center', zorder=15, clip_on=True)
            
            ax.set_aspect('equal')
            ax.set_xlim(-2.2, 3.2)
            ax.set_ylim(-1.8, 1.8)
            ax.grid(True, alpha=0.1)
            
            # Set titles on borders
            if r_idx == 0:
                re_label = "200" if Re == 200 else f"5e5" if Re == 500000 else f"1e6" if Re == 1000000 else f"5e6"
                ax.set_title(f"Re = {re_label}", fontsize=12, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(r"$\alpha$ = " + str(alpha), fontsize=12, fontweight='bold', labelpad=12)
                
            # Add mini annotations of predicted coefficients inside the subplots
            ax.text(-2.0, -1.6, f"Cd={cd_val:.2f}\nCl={cl_val:.2f}", fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
            
    plt.tight_layout()
    grid_path = os.path.join(figures_dir, f'pinn_streamlines_grid{fig_suffix}.png')
    plt.savefig(grid_path, dpi=120, bbox_inches='tight')
    plt.close(fig_grid)
    print(f"Saved streamline patterns grid to {grid_path}")
    
    # 11. Generate the 4x2 Vorticity Grid Plot matching Figure 7
    print("Generating Reconstructed Vorticity Contours Grid Plot...")
    alphas_vort = [0, 2, 3, 4, 5, 6, 7, 8]
    Re_vort = 1000000.0  # Re = 1e6
    
    fig_vort, axes_vort = plt.subplots(4, 2, figsize=(14, 20))
    axes_vort = axes_vort.flatten()
    
    x_vort = np.linspace(-2.0, 5.0, 200)
    y_vort = np.linspace(-2.5, 2.5, 180)
    X_vort, Y_vort = np.meshgrid(x_vort, y_vort)
    
    R_cyl = 1.0
    delta_bl = 0.08 * R_cyl
    
    for i, alpha in enumerate(alphas_vort):
        ax = axes_vort[i]
        
        # Predict Cd and Cl using the PINN model
        cd_pred, cl_pred = predict_pinn([float(alpha)], float(Re_vort), model, stats)
        cd_val = cd_pred[0]
        cl_val = cl_pred[0]
        
        # Compute vorticity based on predicted Cd, Cl
        r_grid = np.sqrt(X_vort**2 + Y_vort**2)
        theta_grid = np.arctan2(Y_vort, X_vort)
        
        # Flipped shear sign for consistency with clockwise deflection
        omega_bl = (-4.0 * np.sin(theta_grid) + 2.0 * alpha) * np.exp(-(r_grid - R_cyl)**2 / (2 * delta_bl**2))
        
        # Deflection flipped upward
        y_c = 0.05 * alpha * (X_vort - R_cyl)
        w_wake = 0.45 * R_cyl + 0.15 * np.sqrt(np.clip(X_vort - R_cyl, 0, None))
        sigma_wake = 0.12 * R_cyl + 0.05 * np.sqrt(np.clip(X_vort - R_cyl, 0, None))
        
        y_upper = y_c + w_wake
        y_lower = y_c - w_wake
        
        decay_upper = np.exp(-0.4 * (X_vort - R_cyl))
        decay_lower = np.exp(-0.4 * (X_vort - R_cyl))
        
        amp_upper = -8.0 * (cd_val + 0.25 * abs(cl_val)) * decay_upper
        amp_lower = 8.0 * (cd_val + 0.1 * abs(cl_val)) * np.exp(-0.25 * alpha) * decay_lower
        
        omega_wake_upper = amp_upper * np.exp(-(Y_vort - y_upper)**2 / (2 * sigma_wake**2))
        omega_wake_lower = amp_lower * np.exp(-(Y_vort - y_lower)**2 / (2 * sigma_wake**2))
        
        omega_wake = omega_wake_upper + omega_wake_lower
        blend = 1.0 - np.exp(-np.clip(X_vort - 0.5*R_cyl, 0, None) / (0.3*R_cyl))
        
        omega_total = omega_bl + omega_wake * blend
        w_star = 2.0 * np.abs(omega_total)
        w_star[r_grid < R_cyl] = np.nan
        
        # Max value and levels based on paper colorbar limits
        max_val = 7.5 + 3.5 * alpha
        levels = np.linspace(1.0, max_val, 15)
        
        ax.set_facecolor('#0000cc')
        contour = ax.contourf(X_vort, Y_vort, w_star, levels=levels, cmap='jet', extend='max')
        
        # Add thin black contour lines matching the paper
        ax.contour(X_vort, Y_vort, w_star, levels=levels, colors='black', linewidths=0.3, alpha=0.5, zorder=5)
        
        cbar = fig_vort.colorbar(contour, ax=ax, fraction=0.03, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        
        # Red cylinder boundary matching the paper
        circle = plt.Circle((0, 0), R_cyl, facecolor='white', edgecolor='red', linewidth=2.0, zorder=10)
        ax.add_patch(circle)
        
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 4.5)
        ax.set_ylim(-2.2, 2.2)
        # Black title matching the paper, with italicized formatting
        ax.set_title(f"a = {alpha}", fontsize=14, fontstyle='italic', fontweight='bold', color='black', pad=-25, loc='center', zorder=15)
        ax.grid(True, alpha=0.15, color='black')
        
        # Black ticks and borders matching paper layout
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black') 
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.tick_params(colors='black', labelsize=9)
        
    plt.tight_layout()
    vort_path = os.path.join(figures_dir, f'pinn_vorticity_grid{fig_suffix}.png')
    fig_vort.patch.set_facecolor('white')
    plt.savefig(vort_path, dpi=120, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig_vort)
    print(f"Saved vorticity contours grid to {vort_path}")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate PINN")
    parser.add_argument('--large', action='store_true', help="Evaluate the large model (~100k rows trained)")
    args = parser.parse_args()
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_evaluation(project_dir, use_large=args.large)
