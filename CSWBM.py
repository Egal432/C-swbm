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
    melting : float, optional
        Snow melting rate parameter (only needed if use_snow=True)
    use_snow : bool
        Whether to use snow module (default: False)
    """
    
    def __init__(self, exp_runoff, exp_et, beta, whc, melting=None, use_snow=False):
        self.exp_runoff = exp_runoff
        self.exp_et = exp_et
        self.beta = beta
        self.whc = whc
        self.melting = melting
        self.use_snow = use_snow

        self.soilm = None
        self.runoff = None
        self.et = None
        self.snow = None
        self.length = None
        self.data = None
        
    def load_data(self, filepath):
        """Load and preprocess ERA5 data from CSV file."""
        raw_data = pd.read_csv(filepath)
        self.data = prepro(raw_data)
        return self.data
    
    def spinup(self, precip, rad, n_years=5):
        """Spin up the model to equilibrium."""
        n_days = min(n_years * 365, len(precip))
        length = len(precip)
        
        soilm = np.full(length, np.nan)
        et = np.full(length, np.nan)
        et_corr = np.full(length, np.nan)
        infiltration = np.full(length, np.nan)
        infiltration_corr = np.full(length, np.nan)
        
        soilm[0] = 0.9 * self.whc
        
        for i in range(1, n_days):
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
        
        # Handle negative radiation (dew)
        negative_rad = rad < 0
        precip[negative_rad] = precip[negative_rad] + (-1) * self.beta * rad[negative_rad]
        rad[negative_rad] = 0
        self.snow = np.zeros(self.length)
        
        # Initialize arrays
        self.soilm = np.full(self.length, np.nan)
        self.runoff = np.full(self.length, np.nan)
        self.et = np.full(self.length, np.nan)
        et_corr = np.full(self.length, np.nan)
        infiltration = np.full(self.length, np.nan)
        infiltration_corr = np.full(self.length, np.nan)
        
        # Spin up
        self.soilm[0] = self.spinup(precip, rad)
        
        # Main model run
        for i in range(1, self.length):
            self.et[i-1] = rad[i-1] * self.beta * min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_et)
            et_corr[i-1] = (rad[i-1] * self.beta * 
                           min(max(0.0, self.whc - self.soilm[i-1]), self.exp_et / self.whc) * 
                           (self.soilm[i-1] / self.whc) ** (self.exp_et - 1.0))
            infiltration[i-1] = (1.0 - min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            infiltration_corr[i-1] = ((-1.0) * 
                                     min(max(0.0, self.whc - self.soilm[i-1]), self.exp_runoff / self.whc) * 
                                     (self.soilm[i-1] / self.whc) ** (self.exp_runoff - 1.0) * 
                                     precip[i-1])
            self.et[i-1] = min(self.et[i-1], self.soilm[i-1] - 5.0)
            self.soilm[i] = self.soilm[i-1] + ((infiltration[i-1] - self.et[i-1]) / 
                                               (1.0 + et_corr[i-1] - infiltration_corr[i-1]))
            self.runoff[i-1] = (min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            self.et[i-1] = self.et[i-1] + (self.soilm[i] - self.soilm[i-1]) * et_corr[i-1]
        
        return self.get_results()
    
    def get_results(self):
        """Return model results as a dictionary."""
        return {
            'soilmoisture': self.soilm,
            'runoff': self.runoff,
            'evapotranspiration': self.et,
            'snow': self.snow,
            'exp_runoff': self.exp_runoff,
            'exp_et': self.exp_et,
            'beta': self.beta,
            'whc': self.whc,
            'melting': self.melting,
            'length': self.length
        }


