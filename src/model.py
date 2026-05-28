import tensorflow as tf
from tensorflow import keras

def build_pinn(hidden_units=128, n_layers=5):
    """
    Builds the Physics-Informed Neural Network architecture using Keras.
    
    Inputs:
        alpha_norm, Re_norm (normalized spin ratio and Reynolds number)
    Outputs:
        Cd_norm, Cl_norm (normalized drag and lift coefficients)
    """
    # Define inputs
    inputs = keras.Input(shape=(2,), name='alpha_Re')
    
    x = inputs
    # Add hidden layers with tanh activation and Glorot normal initialization
    for i in range(n_layers):
        x = keras.layers.Dense(
            hidden_units,
            activation='tanh',
            kernel_initializer='glorot_normal',
            bias_initializer='zeros',
            name=f'hidden_{i+1}'
        )(x)
        
    # Output layer (2 units: Cd and Cl)
    outputs = keras.layers.Dense(
        2,
        activation='linear',
        kernel_initializer='glorot_normal',
        bias_initializer='zeros',
        name='Cd_Cl'
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='PINN_Model')
    return model

if __name__ == '__main__':
    # Print model summary to verify architecture
    model = build_pinn()
    model.summary()
