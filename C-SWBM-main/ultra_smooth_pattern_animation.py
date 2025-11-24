import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from CSWBM import SimpleWaterBalanceModel, prepro

def ultra_smooth_pattern_animation(filepath, base_params, param_configs, save_path=None, fps=30, ylim_dict=None):
    """
    Ultra smooth animation with base→max→min→base pattern for each parameter
    """
    raw_data = pd.read_csv(filepath)
    data = prepro(raw_data)
    dates = data['time']
    
    # Generate smooth sequences with base→max→min→base pattern
    param_sequences = {}
    param_names = list(param_configs.keys())
    
    for param_name, config in param_configs.items():
        base_val = base_params[param_name]
        min_val = config['min']
        max_val = config['max']
        steps = config['steps']
        
        # Create four segments for smooth transition:
        # 1. Base → Max
        segment1 = np.linspace(base_val, max_val, steps)
        # 2. Max → Min  
        segment2 = np.linspace(max_val, min_val, steps)[1:]  # Skip duplicate max point
        # 3. Min → Base
        segment3 = np.linspace(min_val, base_val, steps)[1:]  # Skip duplicate min point
        
        # Combine all segments
        full_sequence = np.concatenate([segment1, segment2, segment3])
        
        param_sequences[param_name] = full_sequence
    
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
    print(f"Pattern: base → max → min → base for each parameter")
    
    # Create clean 3-panel layout
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # If ylim_dict not provided, use reasonable defaults
    if ylim_dict is None:
        ylim_dict = {
            'soilmoisture': (0, 800),
            'runoff': (0, 50),  
            'evapotranspiration': (0, 10)
        }
    
    # Store current parameter info for smooth transitions
    current_param_idx = 0
    segment_info = ""
    
    def animate(frame_num):
        nonlocal current_param_idx, segment_info
        
        param_name, param_value, params = all_frames[frame_num]
        
        # Update current parameter index
        new_param_idx = param_names.index(param_name)
        if new_param_idx != current_param_idx:
            current_param_idx = new_param_idx
            print(f"Transitioning to parameter: {param_name}")
        
        # Determine which segment we're in for this parameter
        sequence = param_sequences[param_name]
        base_val = base_params[param_name]
        min_val = param_configs[param_name]['min']
        max_val = param_configs[param_name]['max']
        
        current_idx_in_sequence = np.where(sequence == param_value)[0][0]
        total_segments = len(sequence)
        
        # Determine current segment for informative display
        if param_value == base_val:
            segment_info = "● Base value"
        elif current_idx_in_sequence < len(sequence) // 3:
            segment_info = "↑ Base → Max"
        elif current_idx_in_sequence < 2 * len(sequence) // 3:
            segment_info = "↓ Max → Min" 
        else:
            segment_info = "↑ Min → Base"
        
        # Run model
        model = SimpleWaterBalanceModel(**params)
        results = model.run(data=data)
        
        # Clear axes
        for ax in axes:
            ax.clear()
        
        # Create informative title with segment info
        param_status = []
        for p_name in param_names:
            if p_name == param_name:
                param_status.append(f"→ {p_name.upper()}: {param_value:.3f}")
            else:
                param_status.append(f"{p_name}: {base_params[p_name]}")
        
        title = f"Germany - Parameter Sensitivity Analysis\n" + " | ".join(param_status) + f"\n{segment_info}"
        fig.suptitle(title, fontsize=13)
        
        # Plot soil moisture with FIXED y-limits
        axes[0].plot(dates, results['soilmoisture'], 'b-', linewidth=2.5, alpha=0.9)
        axes[0].set_ylabel('Soil Moisture\n(mm)', fontsize=11, rotation=0, ha='right')
        axes[0].set_ylim(ylim_dict['soilmoisture'])
        axes[0].grid(True, alpha=0.2)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot runoff with FIXED y-limits
        axes[1].plot(dates, results['runoff'], 'r-', linewidth=2.5, alpha=0.9)
        axes[1].set_ylabel('Runoff\n(mm)', fontsize=11, rotation=0, ha='right')
        axes[1].set_ylim(ylim_dict['runoff'])
        axes[1].grid(True, alpha=0.2)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Plot ET with FIXED y-limits
        axes[2].plot(dates, results['evapotranspiration'], 'g-', linewidth=2.5, alpha=0.9)
        axes[2].set_ylabel('ET\n(mm)', fontsize=11, rotation=0, ha='right')
        axes[2].set_ylim(ylim_dict['evapotranspiration'])
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].grid(True, alpha=0.2)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return []
    
    # Create animation with specified FPS
    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, 
        interval=1000/fps,
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


# Your configuration
param_configs = {
    'whc': {'min': 300, 'max': 800, 'steps': 100},      # Reduced steps for reasonable length
    'beta': {'min': 0.3, 'max': 1.2, 'steps': 200},     # Reduced steps for reasonable length
    'exp_et': {'min': 0.1, 'max': 1.2, 'steps': 200},   # Reduced steps for reasonable length
    'exp_runoff': {'min': 0.5, 'max': 6, 'steps': 200}  # Reduced steps for reasonable length
}

base_params = {
    'whc': 420,
    'beta': 0.8, 
    'exp_et': 0.5,
    'exp_runoff': 4,
    'melting': 2.0,
    'use_snow': True
}

# Define your y-axis limits
ylim_dict = {
    'soilmoisture': (0, 800),
    'runoff': (0, 50),
    'evapotranspiration': (0, 8)
}

# Create the ultra smooth pattern animation
anim = ultra_smooth_pattern_animation(
    filepath='Data/Data_swbm_Germany_new.csv',
    base_params=base_params,
    param_configs=param_configs,
    save_path='ultra_smooth_pattern_animation.gif',
    fps=24,
    ylim_dict=ylim_dict
)

# Or for the cinematic version with transitions:
# anim = cinematic_smooth_animation(
#     filepath='Data/Data_swbm_Germany_new.csv',
#     base_params=base_params,
#     param_configs=param_configs,
#     save_path='cinematic_smooth_animation.gif',
#     fps=24,
#     ylim_dict=ylim_dict,
#     transition_frames=15
# )