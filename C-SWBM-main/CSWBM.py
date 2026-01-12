"""
Simple Water Balance Model (SWBM)
Based on Orth et al., 2013

This model simulates the water balance of a soil column through:
- Soil moisture dynamics
- Runoff generation 
- Evapotranspiration
"""

import numpy as np
import pandas as pd


def prepro(raw_data):
    """
    Preprocess ERA5 data for SWBM model.
    
    Converts all variables to consistent units (mm for water, datetime for time).
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw input data with columns:
        - time: timestamp
        - latitude, longitude: location
        - snr_[MJ/m2]: surface net radiation
        - tp_[mm]: total precipitation  
        - ro_[m]: runoff
        - sm_[m3/m3]: volumetric soil moisture
        - le_[W/m2]: latent heat flux
        - t2m_[K]: 2m temperature (optional)
    
    Returns
    -------
    pd.DataFrame
        Preprocessed data with columns:
        - time: datetime
        - lat, long: location
        - tp: precipitation [mm]
        - sm: soil moisture [mm] (assumes 1m depth)
        - ro: runoff [mm]
        - le: ET [mm] (converted from latent heat)
        - snr: net radiation [mm water equivalent]
        - temp: temperature [K] (if available)
    
    Notes
    -----
    Conversion factors:
    - Soil moisture: m3/m3 * 1000mm = mm (for 1m depth)
    - Runoff: m * 24000 = mm (assuming instant conversion)
    - Latent heat: W/m2 * (86400s/day) / (2.26e6 J/kg) = mm/day
    - Radiation: MJ/m2 / 2.26 = mm water equivalent
    """
    data = {
        'time': pd.to_datetime(raw_data['time']),
        'lat': raw_data['latitude'],
        'long': raw_data['longitude'],
        'tp': raw_data['tp_[mm]'],
        'sm': raw_data['sm_[m3/m3]'] * 1000,  # m3/m3 -> mm (1m depth)
        'ro': raw_data['ro_[m]'] * 24000,      # m -> mm
        'le': raw_data['le_[W/m2]'] * (86400 / 2260000),  # W/m2 -> mm/day
        'snr': raw_data['snr_[MJ/m2]'] * (1 / 2.26),      # MJ/m2 -> mm
    }
    
    # Add temperature if available
    if 't2m_[K]' in raw_data.columns:
        data['temp'] = raw_data['t2m_[K]']
    
    return pd.DataFrame(data)


class SimpleWaterBalanceModel:
    """
    Simple Water Balance Model based on Orth et al. (2013).
    
    The model uses a bucket approach with:
    - Nonlinear relationships between soil moisture and runoff/ET
    - Implicit numerical solution for stability
    - Spin-up period to reach equilibrium
    
    Parameters
    ----------
    exp_runoff : float
        Runoff exponent (controls nonlinearity of runoff generation)
        Typical range: 1-5, default: 2.0
        Higher values = more threshold behavior (sudden runoff when soil saturates)
        
    exp_et : float  
        ET exponent (controls soil moisture limitation of ET)
        Typical range: 0.1-2.0, default: 0.5
        Higher values = stronger reduction of ET as soil dries
        
    beta : float
        Priestley-Taylor coefficient (relates radiation to potential ET)
        Typical range: 0.5-1.2, default: 0.8
        Higher values = more ET for given radiation
        
    whc : float
        Water holding capacity [mm] (maximum soil moisture storage)
        Typical range: 50-300mm, default: 150mm
        Depends on soil depth and texture
    
    Attributes
    ----------
    soilm : np.ndarray
        Simulated soil moisture time series [mm]
    runoff : np.ndarray
        Simulated runoff time series [mm/day]
    et : np.ndarray
        Simulated evapotranspiration time series [mm/day]
    data : pd.DataFrame
        Loaded input data
    length : int
        Length of simulation period
        
    References
    ----------
    Orth, R., et al. (2013). Inferring soil moisture memory from streamflow 
    observations using a simple water balance model. Journal of Hydrometeorology.
    """
    
    def __init__(self, exp_runoff, exp_et, beta, whc, alpha):
        # Model parameters
        self.exp_runoff = exp_runoff
        self.exp_et = exp_et
        self.beta = beta
        self.whc = whc
        self.alpha = alpha
        
        # Model state variables (initialized after run)
        self.k_gw = 0.05
        self.soilm = None
        self.runoff = None
        self.et = None
        self.length = None
        self.data = None
        self.gw_storage = None
        self.baseflow = None
        
    def load_data(self, filepath):
        """
        Load and preprocess ERA5 data from CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file with ERA5 data
            
        Returns
        -------
        pd.DataFrame
            Preprocessed data ready for model
        """
        print(f"    Loading: {filepath}")
        raw_data = pd.read_csv(filepath)
        self.data = prepro(raw_data)
        print(f"    ✓ Preprocessed {len(self.data)} days")
        return self.data
    
    def spinup(self, precip, rad, n_years=5):
        """
        Spin up the model to reach equilibrium soil moisture.
        
        Runs the model for n_years and returns the mean soil moisture
        on December 31st to use as initial condition.
        
        Parameters
        ----------
        precip : np.ndarray
            Precipitation time series [mm]
        rad : np.ndarray
            Net radiation time series [mm water equivalent]
        n_years : int, optional
            Number of years to spin up (default: 5)
            
        Returns
        -------
        float
            Equilibrium soil moisture [mm] to use as initial condition
        """
        n_days = min(n_years * 365, len(precip))
        length = len(precip)
        
        # Initialize arrays
        soilm = np.full(length, np.nan)
        et = np.full(length, np.nan)
        et_corr = np.full(length, np.nan)
        infiltration = np.full(length, np.nan)
        infiltration_corr = np.full(length, np.nan)
        
        # Start at 90% of capacity
        soilm[0] = 0.9 * self.whc
        
        # Run spin-up period
        for i in range(1, n_days):
            # Calculate ET (Eq. 2 in Orth et al., 2013)
            et[i-1] = rad[i-1] * self.beta * min(1.0, (soilm[i-1] / self.whc) ** self.exp_et)
            
            # Calculate ET derivative w.r.t. soil moisture (for implicit solution)
            et_corr[i-1] = (rad[i-1] * self.beta * 
                           min(max(0.0, self.whc - soilm[i-1]), self.exp_et / self.whc) * 
                           (soilm[i-1] / self.whc) ** (self.exp_et - 1.0))
            
            # Calculate infiltration (P-Q) (Eq. 3 in Orth et al., 2013)
            infiltration[i-1] = (1.0 - min(1.0, (soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            
            # Calculate infiltration derivative
            infiltration_corr[i-1] = ((-1.0) * 
                                     min(max(0.0, self.whc - soilm[i-1]), self.exp_runoff / self.whc) * 
                                     (soilm[i-1] / self.whc) ** (self.exp_runoff - 1.0) * 
                                     precip[i-1])
            
            # Limit ET to available water (keep 5mm for wilting point)
            et[i-1] = min(et[i-1], soilm[i-1] - 5.0)
            
            # Implicit solution (Eq. 7 in Orth et al., 2013)
            # More stable than explicit forward Euler
            soilm[i] = soilm[i-1] + ((infiltration[i-1] - et[i-1]) / 
                                     (1.0 + et_corr[i-1] - infiltration_corr[i-1]))
        
        # Return mean soil moisture on Dec 31st of each year
        dec31_indices = []
        for year in range(n_years):
            idx = (year + 1) * 365 - 1
            if idx < n_days and not np.isnan(soilm[idx]):
                dec31_indices.append(idx)
        
        if dec31_indices:
            return np.mean([soilm[idx] for idx in dec31_indices])
        else:
            return 0.9 * self.whc
    
    def run(self, data=None, precip=None, rad=None):
        """
        Run the water balance model.
        
        Can be called in three ways:
        1. After load_data(): model.run() - uses self.data
        2. With DataFrame: model.run(data=df)
        3. With arrays: model.run(precip=p, rad=r)
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Preprocessed data from prepro() function
        precip : np.ndarray, optional
            Precipitation data [mm]
        rad : np.ndarray, optional
            Radiation data [mm water equivalent]
            
        Returns
        -------
        dict
            Model results (see get_results() for details)
            
        Raises
        ------
        ValueError
            If no data is provided through any method
        """
        # Priority: explicit data > stored data > explicit arrays
        if data is not None:
            self.data = data
            precip = data['tp'].values.copy()
            rad = data['snr'].values.copy()
        elif self.data is not None:
            # Use previously loaded data
            precip = self.data['tp'].values.copy()
            rad = self.data['snr'].values.copy()
        elif precip is not None and rad is not None:
            # Use provided arrays
            precip = precip.copy()
            rad = rad.copy()
        else:
            raise ValueError(
                "Must provide data! Use one of:\n"
                "  1. model.load_data('file.csv') then model.run()\n"
                "  2. model.run(data=dataframe)\n"
                "  3. model.run(precip=array, rad=array)"
            )
        
        self.length = len(precip)
        
        # Initialize output arrays
        self.soilm = np.full(self.length, np.nan)
        self.runoff = np.full(self.length, np.nan)
        self.et = np.full(self.length, np.nan)
        et_corr = np.full(self.length, np.nan)
        infiltration = np.full(self.length, np.nan)
        infiltration_corr = np.full(self.length, np.nan)
        
        # Get initial condition from spin-up
        print("    Running spin-up...")
        self.soilm[0] = self.spinup(precip, rad)
        print(f"    Initial soil moisture: {self.soilm[0]:.1f} mm")
        
        # Main model loop
        print("    Running main simulation...")
        for i in range(1, self.length):
            # Calculate ET (limited by soil moisture)
            self.et[i-1] = rad[i-1] * self.beta * min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_et)
            
            # ET derivative
            et_corr[i-1] = (rad[i-1] * self.beta * 
                           min(max(0.0, self.whc - self.soilm[i-1]), self.exp_et / self.whc) * 
                           (self.soilm[i-1] / self.whc) ** (self.exp_et - 1.0))
            
            # Calculate infiltration (precipitation minus runoff)
            infiltration[i-1] = (1.0 - min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            
            # Infiltration derivative
            infiltration_corr[i-1] = ((-1.0) * 
                                     min(max(0.0, self.whc - self.soilm[i-1]), self.exp_runoff / self.whc) * 
                                     (self.soilm[i-1] / self.whc) ** (self.exp_runoff - 1.0) * 
                                     precip[i-1])
            
            # Limit ET (permanent wilting point at 5mm)
            self.et[i-1] = min(self.et[i-1], self.soilm[i-1] - 5.0)
            
            # Update soil moisture (implicit solution)
            self.soilm[i] = self.soilm[i-1] + ((infiltration[i-1] - self.et[i-1]) / 
                                               (1.0 + et_corr[i-1] - infiltration_corr[i-1]))
            
            # Calculate runoff (precipitation minus infiltration)
            self.runoff[i-1] = (min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            
            # Final ET correction
            self.et[i-1] = self.et[i-1] + (self.soilm[i] - self.soilm[i-1]) * et_corr[i-1]
        
        print("    ✓ Simulation complete")
        return self.get_results()
    
    def get_results(self):
        """
        Return model results as a dictionary.
        
        Returns
        -------
        dict
            Dictionary containing:
            - soilmoisture: np.ndarray [mm]
            - runoff: np.ndarray [mm/day]
            - evapotranspiration: np.ndarray [mm/day]
            - exp_runoff: float (parameter)
            - exp_et: float (parameter)
            - beta: float (parameter)
            - whc: float [mm] (parameter)
            - length: int (number of days)
        """
        return {
            'soilmoisture': self.soilm,
            'runoff': self.runoff,
            'evapotranspiration': self.et,
            'exp_runoff': self.exp_runoff,
            'exp_et': self.exp_et,
            'beta': self.beta,
            'whc': self.whc,
            'length': self.length
        }
