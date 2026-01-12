import numpy as np
import pandas as pd


def prepro(raw_data):
    """ Preprocess data for SWBM
    Convert runoff, latent heat flux and solar net radiation to mm.
    Convert time to date.

    :param raw_data: raw input data (pandas df):
         -snr: surface net radiation
         -tp: total precipitation
         -ro: runoff
         -sm: soil moisture at the surface
         -le: latent heat flux
    :return: pre-processed data (pandas df)
    """
    data = {'time': pd.to_datetime(raw_data['time']),
            'lat': raw_data['latitude'],
            'long': raw_data['longitude'],
            'tp': raw_data['tp_[mm]'],
            'sm': raw_data['sm_[m3/m3]'] * 1000,
            'ro': raw_data['ro_[m]'] * 24000,
            'le': raw_data['le_[W/m2]'] * (86400 / 2260000),
            'snr': raw_data['snr_[MJ/m2]'] * (1 / 2.26),
            }
    return pd.DataFrame(data)


class SimpleWaterBalanceModel:
    """
    Simple water balance model based on Orth et al., 2013.
    Extended with groundwater storage component for improved runoff simulation.
    
    Parameters
    ----------
    exp_runoff : float
        Runoff exponent parameter
    exp_et : float
        Evapotranspiration exponent parameter
    beta : float
        Beta parameter for ET calculation
    whc : float
        Water holding capacity (mm)
    alpha : float
        Fast runoff fraction (0-1). Represents the fraction of excess water
        that becomes immediate surface runoff. The remainder (1-alpha) percolates
        to groundwater storage and contributes to baseflow.
        Typical values: 0.2-0.4 (default: 0.3)
    """
    
    def __init__(self, exp_runoff, exp_et, beta, whc, alpha=0.3):
        self.exp_runoff = exp_runoff
        self.exp_et = exp_et
        self.beta = beta
        self.whc = whc
        self.alpha = alpha  # NEW: fast runoff fraction
        
        # Fixed groundwater recession coefficient
        # k_gw = 0.05 means ~14 day half-life (ln(2)/0.05 ≈ 14)
        self.k_gw = 0.05  # FIXED: typical catchment recession rate

        # State variables
        self.soilm = None
        self.runoff = None
        self.et = None
        self.gw_storage = None  # NEW: groundwater storage
        self.baseflow = None    # NEW: baseflow component
        self.fast_runoff = None # NEW: fast runoff component
        self.length = None
        self.data = None
        
    def load_data(self, filepath):
        """Load and preprocess ERA5 data from CSV file."""
        raw_data = pd.read_csv(filepath)
        self.data = prepro(raw_data)
        return self.data
    
    def spinup(self, precip, rad, n_years=5):
        """
        Spin up the model to equilibrium.
        Now includes groundwater storage spinup.
        """
        n_days = min(n_years * 365, len(precip))
        length = len(precip)
        
        # Soil moisture arrays
        soilm = np.full(length, np.nan)
        et = np.full(length, np.nan)
        et_corr = np.full(length, np.nan)
        infiltration = np.full(length, np.nan)
        infiltration_corr = np.full(length, np.nan)
        
        # Groundwater arrays
        gw_storage = np.full(length, np.nan)
        baseflow = np.full(length, np.nan)
        
        # Initialize
        soilm[0] = 0.9 * self.whc
        gw_storage[0] = 0.0
        baseflow[0] = 0.0
        
        for i in range(1, n_days):
            # Soil moisture calculations (existing)
            et[i-1] = rad[i-1] * self.beta * min(1.0, (soilm[i-1] / self.whc) ** self.exp_et)
            et_corr[i-1] = (rad[i-1] * self.beta * 
                           min(max(0.0, self.whc - soilm[i-1]), self.exp_et / self.whc) * 
                           (soilm[i-1] / self.whc) ** (self.exp_et - 1.0))
            infiltration[i-1] = (1.0 - min(1.0, (soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            infiltration_corr[i-1] = ((-1.0) * 
                                     min(max(0.0, self.whc - soilm[i-1]), self.exp_runoff / self.whc) * 
                                     (soilm[i-1] / self.whc) ** (self.exp_runoff - 1.0) * 
                                     precip[i-1])
            et[i-1] = min(et[i-1], soilm[i-1] - 5.0)
            soilm[i] = soilm[i-1] + ((infiltration[i-1] - et[i-1]) / 
                                     (1.0 + et_corr[i-1] - infiltration_corr[i-1]))
            
            # Groundwater calculations (NEW)
            excess = (min(1.0, (soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            percolation = (1.0 - self.alpha) * excess
            gw_storage[i] = gw_storage[i-1] + percolation - baseflow[i-1]
            baseflow[i] = self.k_gw * gw_storage[i]
        
        # Return initial conditions from December 31st values
        dec31_indices = []
        for year in range(n_years):
            idx = (year + 1) * 365 - 1
            if idx < n_days and not np.isnan(soilm[idx]):
                dec31_indices.append(idx)
        
        if dec31_indices:
            soilm_init = np.mean([soilm[idx] for idx in dec31_indices])
            gw_init = np.mean([gw_storage[idx] for idx in dec31_indices])
            return soilm_init, gw_init
        else:
            return 0.9 * self.whc, 0.0
    
    def run(self, data=None, precip=None, rad=None):
        """
        Run the water balance model with groundwater component.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Preprocessed data from prepro() function
        precip : array, optional
            Precipitation data (mm)
        rad : array, optional
            Radiation data (mm water equivalent)
        """
        if data is not None:
            self.data = data
            precip = data['tp'].values.copy()
            rad = data['snr'].values.copy()
        
        if precip is None or rad is None:
            raise ValueError("Must provide either data or (precip, rad)")
        
        precip = precip.copy()
        rad = rad.copy()
        
        self.length = len(precip)
        
        # Initialize arrays
        self.soilm = np.full(self.length, np.nan)
        self.runoff = np.full(self.length, np.nan)
        self.et = np.full(self.length, np.nan)
        self.gw_storage = np.full(self.length, np.nan)  # NEW
        self.baseflow = np.full(self.length, np.nan)    # NEW
        self.fast_runoff = np.full(self.length, np.nan) # NEW
        
        et_corr = np.full(self.length, np.nan)
        infiltration = np.full(self.length, np.nan)
        infiltration_corr = np.full(self.length, np.nan)
        
        # Spin up and get initial conditions
        soilm_init, gw_init = self.spinup(precip, rad)
        self.soilm[0] = soilm_init
        self.gw_storage[0] = gw_init
        self.baseflow[0] = self.k_gw * gw_init
        self.fast_runoff[0] = 0.0
        
        # Main model run
        for i in range(1, self.length):
            # ET calculations (existing)
            self.et[i-1] = rad[i-1] * self.beta * min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_et)
            et_corr[i-1] = (rad[i-1] * self.beta * 
                           min(max(0.0, self.whc - self.soilm[i-1]), self.exp_et / self.whc) * 
                           (self.soilm[i-1] / self.whc) ** (self.exp_et - 1.0))
            
            # Infiltration calculations (existing)
            infiltration[i-1] = (1.0 - min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            infiltration_corr[i-1] = ((-1.0) * 
                                     min(max(0.0, self.whc - self.soilm[i-1]), self.exp_runoff / self.whc) * 
                                     (self.soilm[i-1] / self.whc) ** (self.exp_runoff - 1.0) * 
                                     precip[i-1])
            
            self.et[i-1] = min(self.et[i-1], self.soilm[i-1] - 5.0)
            
            # Update soil moisture (existing)
            self.soilm[i] = self.soilm[i-1] + ((infiltration[i-1] - self.et[i-1]) / 
                                               (1.0 + et_corr[i-1] - infiltration_corr[i-1]))
            
            # NEW: Calculate excess water available for runoff
            excess = (min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            
            # NEW: Split into fast and slow components
            self.fast_runoff[i-1] = self.alpha * excess
            percolation = (1.0 - self.alpha) * excess
            
            # NEW: Update groundwater storage
            self.gw_storage[i] = self.gw_storage[i-1] + percolation - self.baseflow[i-1]
            
            # NEW: Calculate baseflow from groundwater
            self.baseflow[i] = self.k_gw * self.gw_storage[i]
            
            # MODIFIED: Total runoff is now fast + base
            self.runoff[i-1] = self.fast_runoff[i-1] + self.baseflow[i-1]
            
            # ET correction (existing)
            self.et[i-1] = self.et[i-1] + (self.soilm[i] - self.soilm[i-1]) * et_corr[i-1]
        
        return self.get_results()
    
    def get_results(self):
        """Return model results as a dictionary."""
        return {
            'soilmoisture': self.soilm,
            'runoff': self.runoff,
            'evapotranspiration': self.et,
            'gw_storage': self.gw_storage,        # NEW
            'baseflow': self.baseflow,            # NEW
            'fast_runoff': self.fast_runoff,      # NEW
            'exp_runoff': self.exp_runoff,
            'exp_et': self.exp_et,
            'beta': self.beta,
            'whc': self.whc,
            'alpha': self.alpha,                  # NEW
            'k_gw': self.k_gw,                    # NEW
            'length': self.length
        }
