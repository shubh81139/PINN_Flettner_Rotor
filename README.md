# Physics-Informed Neural Networks (PINN) for Flettner Rotor Aerodynamics
### B.Tech. Thesis in Mechanical Engineering | National Institute of Technology Durgapur

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shubh81139-pinn-flettner-rotor-app-yxnjos.streamlit.app/)

👉 **Live Web App:** [shubh81139-pinn-flettner-rotor-app-yxnjos.streamlit.app](https://shubh81139-pinn-flettner-rotor-app-yxnjos.streamlit.app/)

This repository contains the complete implementation of a surrogate aerodynamic model for a rotating cylinder in cross-flow using **Physics-Informed Neural Networks (PINNs)**. This work replicates and extends the research in Shubham Kumar & Rahul Yadav's Bachelor of Technology Thesis under the supervision of **Dr. Saif Akram** (May 2026).


The project is structured strictly around the **6-Stage Deep Learning Life Cycle** to ensure physical consistency, model safety, and numerical generalization.

---

## 🌀 Repository Structure

```
PINN_Project/
├── data/
│   ├── raw/
│   │   └── literature_data.json         # Raw LES & RANS data points from 9 papers
│   ├── pinn_dataset_multi_re.csv        # Generated augmented training dataset
│   ├── normalization_stats.json         # Normalization scaling metrics
│   ├── train_data.json                  # Scaling-normalized training input/targets
│   └── val_data.json                    # Scaling-normalized validation set
├── figures/                             # Generated thesis validation plots
│   ├── training_history_large.png       # Cosine decay convergence logs
│   ├── pinn_predictions_large.png       # Re=140k prediction curves
│   ├── drag_crisis_large.png            # Drag crisis transition curve
│   ├── flow_regime_map_large.png        # Aerodynamic State & Vortex Map
│   ├── pinn_streamlines_grid_large.png  # Reconstructed streamlines (Figure 6)
│   └── pinn_vorticity_grid_large.png    # Reconstructed vorticity modulus (Figure 7)
├── models/
│   └── pinn_model_large.keras           # Trained TF Keras model weights
├── src/
│   ├── dataset.py                       # Data augmentation, scaling & splitting
│   ├── model.py                         # Dense Neural Network architecture
│   ├── train.py                         # Custom training loop (5,000 epochs)
│   └── evaluate.py                      # Evaluation validation tables & plotting
├── app.py                               # Interactive Streamlit Web App
├── requirements.txt                     # Package dependencies
└── README.md                            # Documentation (this file)
```

---

## 🧠 The 6-Stage Deep Learning Life Cycle

### 1. Problem Understanding
A cylinder rotating in a fluid stream creates velocity differences between the advancing and retreating surfaces due to skin friction drag. This velocity difference generates pressure gradients (via Bernoulli's principle), producing a transverse lift force—the **Magnus Effect**.

The aerodynamic force coefficients are non-dimensionalized as:
$$Cl = \frac{\text{Lift}}{\frac{1}{2}\rho U_{\infty}^2 D H}, \quad Cd = \frac{\text{Drag}}{\frac{1}{2}\rho U_{\infty}^2 D H}$$

These coefficients depend on:
* **Spin Ratio** ($\alpha = \Omega D / 2 U_{\infty}$): Tangential-to-freestream velocity ratio.
* **Reynolds Number** ($Re = U_{\infty} D / \nu$): Ratio of inertial to viscous forces.

Traditional CFD solvers (DNS or LES) take hours to calculate these coefficients. The surrogate model aims to map $(\alpha, Re) \to (Cd, Cl)$ instantaneously.

### 2. Data Collection & Augmentation
Discrete simulation and experimental validation points are collected from literature for 5 Reynolds numbers ($Re \in [60\text{k}, 5\text{M}]$) and spin ratios ($\alpha \in [0, 8]$) (Aoki 2001, Karabelas 2010/2012). To expand the dataset, we execute a **Physics-Informed Augmentation Pipeline** inside [dataset.py](file:///c:/Users/shubh/Desktop/PINN_Project/src/dataset.py):
* **Dense Interpolation**: Fills the continuous $\alpha$ domain.
* **Magnus Symmetry Mirroring**: Teaches the network physical coordinate symmetries:
  $$Cl(-\alpha) = -Cl(\alpha), \quad Cd(-\alpha) = Cd(\alpha)$$
* **Gaussian Noise Addition**: Adds multiple noisy copies ($\sigma_{Cd} = 0.015$, $\sigma_{Cl} = 0.030$) to simulate flow turbulence.

### 3. Data Splitting & Normalization
Due to massive scale differences ($Re \approx 5 \times 10^6$ vs. $\alpha \approx 2.0$), Z-score feature scaling is applied to prevent Reynolds number dominance in gradients:
$$X_{\text{scaled}} = \frac{X - \mu_X}{\sigma_X}, \quad Y_{\text{scaled}} = \frac{Y - \mu_Y}{\sigma_Y}$$
The dataset is then partitioned using a randomized **80% Training Set** and **20% Validation Set** to track model generalization.

### 4. Model Selection (PINN)
A Keras Multilayer Perceptron (MLP) with 5 hidden layers (128 tanh neurons each) and Xavier initialization is selected. To prevent unphysical predictions, we regularize training by incorporating boundary condition losses:
$$L_{\text{total}} = L_{\text{data}} + \lambda \cdot (L_{\text{Magnus}} + L_{\text{zero\_lift}} + L_{\text{drag}} + L_{\text{Prandtl}})$$
1. **Magnus Sign Rule**: Penalizes lift pointing opposite to rotation ($Cl \cdot \alpha < 0$).
2. **Zero-Lift at Rest**: Forces $Cl = 0$ when rotation is zero ($\alpha = 0$).
3. **Drag Positivity**: Penalizes negative drag ($Cd < 0$).
4. **Prandtl Lift Ceiling**: Penalizes lift magnitudes exceeding the potential limit $|Cl| \le 4\pi \approx 12.57$.

### 5. Model Training
Training is executed up to **5,000 Epochs** inside [train.py](file:///c:/Users/shubh/Desktop/PINN_Project/src/train.py) (500 epochs for the large dataset). We implement a **Cosine Decay Learning Rate Schedule** that smoothly decays step size from $\eta_{\text{max}} = 10^{-3}$ to $\eta_{\text{min}} = 10^{-5}$ for optimal convergence:
$$\eta(t) = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

### 6. Model Evaluation
The trained model is evaluated quantitatively by checking Mean Absolute Percentage Error (MAPE) against all five literature datasets (averaging under 1.5% error).

Qualitatively, the predicted lift and drag are used to reconstruct 2D velocity fields (streamlines) and vorticity contours:
* **Apparent Wind Velocity**: $U_{\text{rel}} = \sqrt{V_{\text{wind}}^2 + V_{\text{ship}}^2 + 2 V_{\text{wind}} V_{\text{ship}} \cos(\theta_{\text{wind}})}$
* **Propulsion Thrust**: $F_x = L \sin(\theta_{\text{app}}) - D_f \cos(\theta_{\text{app}})$
* **Spin Power Consumption**: $P_{\text{total, kW}} = \frac{C_M \cdot \pi \cdot \rho \cdot U_{\text{tan}}^3 \cdot \frac{D}{2} \cdot H}{\eta \times 1000}$

---

## 🛠️ Execution Pipeline

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline (Stage 2 & 3)
```bash
python src/dataset.py --large
```

### 3. Run PINN Training (Stage 5)
```bash
python src/train.py --large
```

### 4. Run Model Evaluation & Plotting (Stage 6)
```bash
python src/evaluate.py --large
```

### 5. Launch the Streamlit App
Run the interactive dashboard locally to test predictions, sweeps, and sizing calculations:
```bash
streamlit run app.py
```
