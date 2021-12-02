import pandas as pd
import numpy as np
from typing import List, Dict
from logging import Logger
from mape_maker.datasets.SID import SID
from mape_maker.datasets.XYID import XYID
from mape_maker.utilities.SimParams import SimParams
from mape_maker.datasets.Dataset import load_spatial_data
import gstools as gs
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import norm

class SpatialMapeMaker:
    """A package to simulate actuals or forecasts scenarios for a given mean absolute percent error

    Attributes:
        xyid (:obj:`XYID`): the input dataset with values of actuals and forecasts for specified datetime.
        logger (:obj:`Logger`): the logger to use to display messages to the user during the process
        sid (:obj:`SID`): the simulation input dataset initiated in the function simulate, formatted xyid with the
            exception that one column can be missing for operational purpose.

    """
    __version__ = "0.1"

    def __init__(self, logger: Logger, bus_list_path:str, xyid_path_actuals:str , xyid_path_forecasts: str,
                 ending_feature: str = "actuals",
                 xyid_load_pickle: bool = False,
                 input_start_dt: str = None, input_end_dt: str = None, a: float = 4, base_process="ARMA", scale_by_capacity: float = None) -> None:
        """Init statement of MapeMaker class

        Args:
            logger (Logger): the type of logger you want to use for all the process of MapeMaker.
            xyid_path (str): the filepath to the input data csv.
            ending_feature (str, optional): the feature you wish to simulate (either "actuals" or "forecasts"). Defaults
                to "actuals".
            xyid_load_pickle (bool): if you want to load the pickle file storing the estimated coefficients estimated
                in a previous run. Defaults to False.
            input_start_dt (str): start date of the input data if you want to select a subset of the csv file. Defaults
                to None.
            input_end_dt (str): end date of the input data if you want to select a subset of the csv file. Defaults
                to None.
            a (float): percent of the input dataset to use on the right and left of the conditional distribution
                coefficient estimation.
            scale_by_capacity (float): if you want MAPE to be with respect to the capacity, enter the capacity; if you want
                the capacity to be the max of actuals(x), enter 0. Defaults to None, meaning the MAPE is with respect to actuals

        """
        self.logger = logger
        self.ending_feature = ending_feature

        bus_list = pd.read_csv(bus_list_path)
        data_buses = bus_list[['Substation Latitude', 'Substation Longitude', 'Number']].values
        self.buses_to_lat_long = dict(
            [(int(data_buses[i][2]), (data_buses[i][0], data_buses[i][1])) for i in range(data_buses.shape[0])])

        spatial_df_forecast = load_spatial_data(xyid_path_forecasts)
        spatial_df_actuals = load_spatial_data(xyid_path_actuals)
        self.xyids = self.build_nodal_xyids(spatial_df_forecast, spatial_df_actuals, base_process=base_process,
                               a=a, start_date=input_start_dt,
                               end_date=input_end_dt,  ending_feature=ending_feature, xyid_load_pickle=xyid_load_pickle,
                               scale_by_capacity=scale_by_capacity) #: step 1: load the XYID, estimate distrib and ARMA
        self.buses_in_datasets = list(self.xyids.keys())
        self.long_lat = np.array([self.xyids[self.buses_in_datasets[i]].long_lat for i in range(len(self.xyids))])
        self.spatial_base_process = self.assemble_spatial_base_process()
        #: step 2: create an object SID. Will be initiated for simulation
        self.sids: SID = type('SID', (), {})()

    def assemble_spatial_base_process(self):
        data = [self.xyids[self.buses_in_datasets[i]].arma_process.z_hat.values for i in range(len(self.buses_in_datasets))]
        spatial_base_process = pd.DataFrame(data=data).T
        spatial_base_process.columns = self.buses_in_datasets
        spatial_base_process.index = self.xyids[self.buses_in_datasets[0]].arma_process.z_hat.index
        return spatial_base_process

    def compute_empirical_spatial_variogram(self):
        n = self.spatial_base_process.shape[0]
        values = self.spatial_base_process.values
        for i in range(n):
            print(i)
            field = values[i]
            emp_v = gs.vario_estimate(self.long_lat.T, field, latlon=True, return_counts=True, sampling_size=50)
            if i == 0:
                final_emp = list(emp_v)
            else:
                final_emp[0] = emp_v[0]
                final_emp[1] += emp_v[1]
                final_emp[2] += emp_v[2]

        final_emp[1] = final_emp[1] / n
        return final_emp

    def fit_kernels(self, plot=False):
        self.spatial_emp = self.compute_empirical_spatial_variogram()
        spatial_kernel = gs.Matern(latlon=True, rescale=gs.EARTH_RADIUS)
        spatial_kernel.fit_variogram(*self.spatial_emp, var=1)

        self.temporal_emp = self.compute_empirical_temporal_variogram()
        temporal_kernel = gs.Matern()
        temporal_kernel.fit_variogram(*self.temporal_emp, var=1)

        self.spatial_kernel = spatial_kernel
        self.temporal_kernel = temporal_kernel
        if plot:
            self.plot_variograms()

        return spatial_kernel, temporal_kernel

    def plot_variograms(self):
        plt.scatter(self.spatial_emp[0], self.spatial_emp[1], label="empirical")
        xs_ = np.linspace(0, 1.5 * np.max(self.spatial_emp[0]), 100)
        plt.plot(xs_, [self.spatial_kernel.variogram(x_) for x_ in xs_], color="red", label="Exponential Kernel")
        plt.title("Empirical Averaged Spatial Semi-variogram of Base Process")
        plt.xlabel("distance ")
        plt.ylabel("Variance")
        plt.legend()
        plt.show()

        plt.scatter(self.temporal_emp[0], self.temporal_emp[1])
        xs_ = np.linspace(0, np.max(self.temporal_emp[0]), 100)
        plt.plot(xs_, [self.temporal_kernel.variogram(x_) for x_ in xs_], color="red", label="Exponential Kernel")
        plt.title("Empirical Averaged Time Semi-variogram of Base Process")
        plt.xlabel("time (hours)")
        plt.ylabel("Variance")
        plt.show()

    def compute_empirical_temporal_variogram(self):
        t = np.array(range(self.spatial_base_process.shape[0]))
        t_bins = np.array(range(1, 20))
        n = self.spatial_base_process.shape[1]
        values = self.spatial_base_process.values
        for i in range(n):
            print(i)
            field = values[:,i]
            emp_v = gs.vario_estimate(t, field, bin_edges=t_bins, latlon=False, return_counts=True, sampling_size=1000)
            if i == 0:
                final_emp = list(emp_v)
            else:
                final_emp[0] = emp_v[0]
                final_emp[1] += emp_v[1]
                final_emp[2] += emp_v[2]

        final_emp[1] = final_emp[1] / n
        return final_emp

    def build_nodal_xyids(self, spatial_df_forecast, spatial_df_actuals, start_date: str = None, end_date: str = None,
                          ending_feature: str = "actuals", xyid_load_pickle: bool = False,
                          a: float = 4, base_process="ARMA", scale_by_capacity: float = None):
        columns = list(set(list(spatial_df_forecast.columns)) & set(list(spatial_df_actuals.columns)))[:5]
        shared_index = list(set(list(spatial_df_forecast.datetime)) & set(list(spatial_df_actuals.datetime)))
        shared_index.sort()

        datasets = {}
        columns.sort()

        # columns = ["160263_1_Wind"] + columns[:2]
        c = columns[0]
        # columns = ['190111_1_Wind'] + columns
        # print(columns[:1])
        for k, c in enumerate(columns):
            if "_" in c:
                print(c, 100 * k / len(columns))
                bus_node_str, _, energy_source = c.split("_")
                bus_node = int(bus_node_str)
                if bus_node in self.buses_to_lat_long:
                    pre_df = pd.DataFrame(index=shared_index, columns=["datetime", "actuals", "forecasts"])
                    pre_df["datetime"] = shared_index
                    pre_df["actuals"] = spatial_df_actuals[c].loc[shared_index]
                    pre_df["forecasts"] = spatial_df_forecast[c].loc[shared_index]

                    # try:
                    node_data = XYID(a, lat_long=self.buses_to_lat_long[bus_node],
                                        base_process=base_process, start_date=start_date,
                                        end_date=end_date, ending_feature=ending_feature, xyid_load_pickle=xyid_load_pickle,
                                        logger=self.logger, scale_by_capacity=scale_by_capacity, df=pre_df, name=c)
                    datasets[bus_node] = node_data
                    # except:
                    #     print("PROBLEM FOR ", c)
                    #     pass
        return datasets

    def build_spatial_covariance(self):
        n = self.long_lat.shape[0]
        C = np.zeros((n, n))
        for j in range(n):
            pos_j = np.array(lon_lat_to_cartesian(*self.long_lat[j]))
            for i in range(j, n):
                pos_i = np.array(lon_lat_to_cartesian(*self.long_lat[i]))
                C[i, j] = C[j, i] = self.spatial_kernel.covariance(np.linalg.norm(pos_j - pos_i) / gs.EARTH_RADIUS)
        self.spatial_C = C
        return C

    def simulate_spatial_base_process(self, start_date, end_date, T=24):
        T = self.spatial_base_process.loc[start_date:end_date].shape[0]
        n = self.long_lat.shape[0]
        bigC = np.zeros((T * n, T * n))

        for t in range(T):
            for t_2 in range(t, T):
                t_cov = self.temporal_kernel.covariance(np.linalg.norm(t - t_2))
                bigC[t * n: (t + 1) * n, t_2 * n: (t_2 + 1) * n] = bigC[t_2 * n: (t_2 + 1) * n,
                                                                   t * n: (t + 1) * n] = t_cov * self.spatial_C

        xs = multivariate_normal(np.zeros(T * n), cov=bigC)

        ts = np.arange(0, T)
        data = np.zeros((T, n))
        for i in range(n):
            data[:, i] = xs[ts * n + i]

        simul_base_process = pd.DataFrame(index=self.spatial_base_process.loc[start_date:end_date].index,
                                          data=data, columns=self.spatial_base_process.columns)

        simul_base_process = simul_base_process.apply(norm.cdf)

        return simul_base_process


    def simulate(self, number_of_simulations=1, sid_file_path: str = None, simulation_start_dt: str = None, simulation_end_dt: str = None,
                 output_dir: str = None, list_of_date_ranges: List[str] = None, seed: int = None, **kwargs) -> pd.DataFrame:
        """simulate scenarios

        Args:
            sid_file_path (str): the filepath to the simulation input data csv. Defaults to None. In that case, the sid
                will be a subset of the xyid.
            simulation_start_dt: start date of the scenarios. Defaults to None. In that case, the start_date is the
                first date of the sid.
            simulation_end_dt: end date of the scenarios. Defaults to None. In that case the end_date is the last
                date of the sid.
            output_dir: the path where to store the results csv. Defaults to None. In that case, no output is
                produced.
            list_of_date_ranges : list of [start_date, end_date] over which to simulate
            seed : random seed
            **kwargs: Simulation parameters. They will be given in input of the Simulation Parameters.

        Returns:
            results (pd.DataFrame): a dataframe with the simulations as columns and sid datetime index as index

        """
        np.random.seed(seed=seed)
        simulated_spatial_base_process = [self.simulate_spatial_base_process(simulation_start_dt, simulation_end_dt) for _ in range(number_of_simulations)]
        # self.create_save_simparams(**kwargs)
        results = [pd.DataFrame(columns=self.spatial_base_process.columns) for _ in range(number_of_simulations)]
        # TODO create multiple SID for corresponding start and end date in list of date_ranges
        if list_of_date_ranges is None:
            self.sids = {}
            for bus in self.buses_in_datasets:
                self.sids[bus] = SID(logger=self.logger, csv_filepath=sid_file_path, dataset=self.xyids[bus], start_date=simulation_start_dt,
                                     end_date=simulation_end_dt, ending_feature=self.ending_feature)  #: initiate the SID from a new csv or from xyid
            #: create a SimParams object and adjust distributions
                # self.sids[bus].number_of_simulations = number_of_simulations
                sim_params = SimParams(self.xyids[bus], self.sids[bus], self.logger, n=number_of_simulations)
                self.sids[bus].set_sim_params(sim_params)
                r = self.sids[bus].simulate_multiple_scenarios(output_dir,
                                                               base_processes=np.array([simulated_spatial_base_process[k][bus].values for k in range(number_of_simulations)]),
                                                               **kwargs)  #: simulate and store the scenarios
                # print(r)
                for k in range(number_of_simulations):
                    results[k][bus] = r[r.columns[k]]

        return results

    def create_save_simparams(self, **kwargs: object) -> None:
        """constructs a SimParams object and save it to the sid

        Args:
            **kwargs: simulation parameters. They will pass to the constructor of SimParams.

        """
        sim_params = SimParams(self.xyid, self.sid, self.logger, **kwargs)
        self.sid.set_sim_params(sim_params)

    def get_results(self) -> List[pd.DataFrame]:
        """returns the list of the different simulations of scenarios

        Returns:

        """
        return self.sid.simulations


def lon_lat_to_cartesian(lon, lat, R = 6371):
    """
    Returns Cartesian coordinates of a point on a sphere with radius R = 6371
    km for earth
    """
    import numpy as np
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x,y,z


# if __name__ == "__main__":
#     import logging
#     from datetime import datetime
#     from mape_maker.utilities.Scenarios import Scenarios
#     logger = logging.getLogger('make-maker')
#     logging.basicConfig(level=logging.INFO, format='%(message)s')
#     mare_embedder = MapeMaker(logger=logger, ending_feature="forecasts", xyid_path="samples/rts_gmlc/wind_operations_example.csv",
#                               input_start_dt=str(
#                                   datetime(year=2020, month=2, day=1, hour=0, minute=0, second=0)),
#                               input_end_dt=str(
#                                   datetime(year=2020, month=8, day=31, hour=23, minute=0, second=0)),
#                               xyid_load_pickle=True)
#     curvature_parameters = [{
#         "MIP": 0.05,
#         "time_limit": 15,
#         "curvature_target": None,
#         "solver": "gurobi",
#     }, None]
#     results = mare_embedder.simulate(n=1,
#                                      simulation_start_dt=str(
#                                          datetime(year=2020, month=11, day=1, hour=0, minute=0, second=0)),
#                                      simulation_end_dt=str(datetime(year=2020, month=11, day=7, hour=0, minute=0, second=0)))
#     from mape_maker.utilities.Scenarios import Scenarios
#     Scenarios(logger=logger, X=mare_embedder.sid.x_t,
#               Y=mare_embedder.sid.y_t,
#               results=results,
#               target_mare=mare_embedder.sid.SimParams.r_tilde,
#               f_mare=mare_embedder.xyid.dataset_info.get("r_m_hat"))
