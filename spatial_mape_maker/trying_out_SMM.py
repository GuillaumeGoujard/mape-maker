import matplotlib.pyplot as plt
from mape_maker.SpatialMapeMaker import SpatialMapeMaker
import numpy as np
import pandas as pd
import logging

"""
The usual parameters of MapeMaker

Note that base process is set to None: no need to fit an ARMA process
 since we are going to replace it by a Gaussian Process
"""
log = logging.getLogger('mape-maker')
log.level = logging.INFO
kwargs = {
    "logger": log,
    "bus_list_path": "Texas7k_20210529_BusData.csv" ,
    "xyid_path_actuals":"timeseries_data_files/WIND/REAL_TIME_wind.csv",
    "xyid_path_forecasts": "timeseries_data_files/WIND/DAY_AHEAD_wind.csv",
    "input_start_dt": '2018-01-01 00:00:00',
    "input_end_dt": '2018-12-31 23:00:00',
    "a":3,
    "base_process":None,
    "xyid_load_pickle":True,
    "ending_feature":"actuals"
}

smm = SpatialMapeMaker(**kwargs)

"""
fit_kernels estimate the marginal empirical variograms and
fit 2 matern kernels w.r.t to space and time. Look at the plots!
"""
smm.fit_kernels(plot=True)

"""
Assuming a sum of the kernel to be the spatio-temporal kernel, the covariance matrix is easy to build. 
"""
smm.build_spatial_covariance()

"""
Simulate 1 actuals scenarios for December 31st.
"""
print("END OF ESTIMATION")
number_of_simulations = 1
results = smm.simulate(number_of_simulations=number_of_simulations,
                       simulation_start_dt='2018-12-31 00:00:00',
                       simulation_end_dt='2018-12-31 23:00:00')

# CHOLESKY DECOMPOSITION?

"""
The usual plot for one bus
"""
from mape_maker.utilities.Scenarios import Scenarios
bus = 150528
s = Scenarios(logger=smm.logger, X=smm.sids[bus].x_t,
      Y=smm.sids[bus].y_t,
      results=pd.DataFrame(index=results[0][bus].index, data=np.array([results[k][bus].values for k in range(number_of_simulations)]).T),
      target_mare=1,
      f_mare=1)
plt.show()


