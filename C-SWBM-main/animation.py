import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from CSWBM import SimpleWaterBalanceModel, prepro  # Replace with your actual filename

def create_model_animation(filepath, param_name, param_range, fixed_params):
    """
    Create animation showing how changing parameters affect model outputs
    
    Args:
        filepath: Path to Germany CSV file
        param_name: Parameter to animate ('whc', 'beta', 'exp_et', 'exp_runoff')
        param_range: Array of parameter values
        fixed_params: Dictionary of fixed parameters
    """
    
    # Load data once (more efficient)
    raw_data = pd.read_csv(filepath)
    data = prepro(raw_data)
    dates = data['time']
    
    # Initialize model with first parameter set
    initial_params = fixed_params.copy()
    initial_params[param_name] = param_range[0]
    model = SimpleWaterBalanceModel(**initial_params)
    
    # Run initial model to get data structure
    initial_results = model.run(data=data)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Germany - Parameter Sensitivity: {param_name.upper()}', fontsize=14)
    
    # Store lines for updating
    lines = []
    
    def animate(frame_num):
        param_value = param_range[frame_num]
        
        # Update parameter and run model
        current_params = fixed_params.copy()
        current_params[param_name] = param_value
        
        model = SimpleWaterBalanceModel(**current_params)
        results = model.run(data=data)
        
        # Clear axes
        for ax in [ax1, ax2, ax3]:
            ax.clear()
        
        # Plot soil moisture
        ax1.plot(dates, results['soilmoisture'], 'b-', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('Soil Moisture (mm)')
        ax1.set_title(f'Soil Moisture | {param_name} = {param_value:.3f}')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot runoff
        ax2.plot(dates, results['runoff'], 'r-', linewidth=1.5, alpha=0.8)
        ax2.set_ylabel('Runoff (mm)')
        ax2.set_title('Runoff')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot evapotranspiration
        ax3.plot(dates, results['evapotranspiration'], 'g-', linewidth=1.5, alpha=0.8)
        ax3.set_ylabel('ET (mm)')
        ax3.set_title('Evapotranspiration')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return []
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(param_range), 
        interval=600, repeat=True, blit=False
    )
    
    plt.show()
    return anim

# Usage example
fixed_params = {
    'whc': 420.0,
    'beta': 0.8, 
    'exp_et': 0.5,
    'exp_runoff': 4,
    'melting': 2.0,  # Added melting parameter
    'use_snow': True
}


# Animate WHC parameter
filepath = 'Data/Data_swbm_Germany_new.csv'
whc_range = np.linspace(200, 800, 480)
anim = create_model_animation(filepath, 'whc', whc_range, fixed_params)

# Save if desired
anim.save('whc_animation.gif', writer='pillow', fps=24)
