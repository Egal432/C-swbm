import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from CSWBM import SimpleWaterBalanceModel, prepro  # Replace with your actual filename

def ultra_smooth_yoyo_animation(filepath, base_params, param_configs, save_path=None, fps=30):
    """
    Ultra smooth animation with maximum steps and high FPS
    """
    raw_data = pd.read_csv(filepath)
    data = prepro(raw_data)
    dates = data['time']
    
    # Generate yoyo sequences with maximum smoothness
    param_sequences = {}
    param_names = list(param_configs.keys())
    
    for param_name, config in param_configs.items():
        min_val = config['min']
        max_val = config['max']
        steps = config['steps']
        
        # Create smooth sequence with more points
        up_sequence = np.linspace(min_val, max_val, steps)
        down_sequence = np.linspace(max_val, min_val, steps)[1:-1]
        yoyo_sequence = np.concatenate([up_sequence, down_sequence])
        
        param_sequences[param_name] = yoyo_sequence
    
    # Create combined sequence
    all_frames = []
    
    for param_name in param_names:
        sequence = param_sequences[param_name]
        for value in sequence:
            params = base_params.copy()
            params[param_name] = value
            all_frames.append((param_name, value, params))
    
    total_frames = len(all_frames)
    print(f"Total animation frames: {total_frames}")
    print(f"Animation duration: {total_frames/fps:.1f} seconds at {fps} FPS")
    
    # Create clean 3-panel layout
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    def animate(frame_num):
        param_name, param_value, params = all_frames[frame_num]
        
        # Run model
        model = SimpleWaterBalanceModel(**params)
        results = model.run(data=data)
        
        # Clear axes
        for ax in axes:
            ax.clear()
        
        # Create informative title
        param_status = []
        for p_name in param_names:
            if p_name == param_name:
                param_status.append(f"→ {p_name.upper()}: {param_value:.3f}")
            else:
                param_status.append(f"{p_name}: {base_params[p_name]}")
        
        title = f"Germany - Parameter Sensitivity Analysis\n" + " | ".join(param_status)
        fig.suptitle(title, fontsize=13)
        
        # Plot soil moisture
        axes[0].plot(dates, results['soilmoisture'], 'b-', linewidth=2.5, alpha=0.9)
        axes[0].set_ylabel('Soil Moisture\n(mm)', fontsize=11, rotation=0, ha='right')
        axes[0].grid(True, alpha=0.2)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot runoff
        axes[1].plot(dates, results['runoff'], 'r-', linewidth=2.5, alpha=0.9)
        axes[1].set_ylabel('Runoff\n(mm)', fontsize=11, rotation=0, ha='right')
        axes[1].grid(True, alpha=0.2)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Plot ET
        axes[2].plot(dates, results['evapotranspiration'], 'g-', linewidth=2.5, alpha=0.9)
        axes[2].set_ylabel('ET\n(mm)', fontsize=11, rotation=0, ha='right')
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].grid(True, alpha=0.2)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return []
    
    # Create animation with specified FPS
    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, 
        interval=1000/fps,  # Dynamic interval based on FPS
        repeat=True, 
        blit=False
    )
    
    # Save animation if path provided
    if save_path:
        print(f"Saving animation to {save_path} at {fps} FPS...")
        try:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps, dpi=120)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=fps, dpi=120, bitrate=2500)
            else:
                anim.save(save_path + '.gif', writer='pillow', fps=fps, dpi=120)
            print("Animation saved successfully!")
        except Exception as e:
            print(f"Error saving animation: {e}")
    
    plt.show()
    return anim

# Ultra smooth configuration
param_configs = {
    'whc': {'min': 300, 'max': 800, 'steps': 100},      # Even more steps
    'beta': {'min': 0.3, 'max': 1.2, 'steps': 200},     # Even more steps
    'exp_et': {'min': 0.1, 'max': 1.2, 'steps': 200},   # Even more steps
    'exp_runoff': {'min': 0.5, 'max': 6, 'steps': 200}    # Even more steps
}

base_params = {
    'whc': 420,
    'beta': 0.8, 
    'exp_et': 0.5,
    'exp_runoff': 4,
    'melting': 2.0,
    'use_snow': True
}

# Create ultra smooth animation
anim = ultra_smooth_yoyo_animation(
    filepath='Data/Data_swbm_Germany_new.csv',
    base_params=base_params,
    param_configs=param_configs,
    save_path='ultra_smooth_animation_bigger_range.gif',
    fps=24  # or 30 for even smoother
)
