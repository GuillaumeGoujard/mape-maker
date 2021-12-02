import matplotlib.pyplot as plt
from mape_maker.SpatialMapeMaker import SpatialMapeMaker
import numpy as np
import pandas as pd
import logging

"""
The usual parameters of MapeMaker

Note that base process is set to None: no need to fit an ARMA process since we are going to replace it by a Gaussian Process
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

# bus = smm.buses_in_datasets[-1]
smm.sids[bus] = SID(logger=smm.logger, csv_filepath=None, dataset=smm.xyids[bus],
                     start_date='2018-12-31 00:00:00',
                     end_date='2018-12-31 23:00:00', ending_feature=smm.ending_feature)
#
sim_params = SimParams(smm.xyids[bus], smm.sids[bus], smm.logger, n=1)

self = sim_params

x = smm.sids[bus].dataset_x[1]
(a, b, loc_nx, scale_nx), dirac_proba_2 = smm.sids[bus].s_x[self.cfx[x]]
loc_nx = -x if loc_nx < -x else loc_nx
scale_nx = self.cap if scale_nx > self.cap else scale_nx
oh = max(abs(loc_nx), abs(scale_nx), self.m_tilde[x])
# NOTE: the upper bound for s should be cap - x - l
print((loc_nx, scale_nx, dirac_proba_2[0], dirac_proba_2[1]))
nl, ns, p0, p1 = least_squares(find_intersections,
                       x0=(loc_nx, scale_nx, dirac_proba_2[0], dirac_proba_2[1]),
                       bounds=([-x, 0, 0, 0], [loc_nx, self.cap, 1, 1]),
                       args=(self.m_tilde[x], a, b, loc_nx, scale_nx, dirac_proba_2, oh, self.cap, False),
                       ftol=1e-3, method="dogbox").x

find_intersections(loc_nx, scale_nx, dirac_proba_2[0], dirac_proba_2[1], *(self.m_tilde[x], a, b, loc_nx, scale_nx, dirac_proba_2, oh, self.cap, False))

find_intersections((loc_nx, scale_nx, dirac_proba_2[0], dirac_proba_2[1]), target=self.m_tilde[x], a=a, b=b, l_hat=loc_nx, s_hat=scale_nx, dirac_proba=dirac_proba_2, oh=oh, cap=self.cap, verbose=False)

x = (loc_nx, scale_nx, dirac_proba_2[0], dirac_proba_2[1])
target = self.m_tilde[smm.sids[bus].dataset_x[1]]
x_bar = smm.sids[bus].dataset_x[1]
l_hat = loc_nx
s_hat = scale_nx
dirac_proba = dirac_proba_2
cap = self.cap
verbose = False
sqarg = x[2]*x_bar + x[3]*(cap-x_bar) + (1 - x[2] - x[3])*integrate_a_mean_2d(x[0], x[1], a=a, b=b, verbose=verbose) - target \
            + abs((x[0] - l_hat) / oh) + abs((x[1] - s_hat) / oh) + abs((x[2] - dirac_proba[0]) / oh) + abs((x[3] - dirac_proba[1]) / oh)


s_x = smm.sids[bus].s_x
dataset_sid = smm.sids[bus].dataset_x

s_x_sid = dict([(key, []) for key in dataset_sid])
(_, _, l, s), dirac_proba = s_x[list(s_x.keys())[0]]
p = len(dataset_sid) // 8
nb_errors = 0
for j, x in enumerate(dataset_sid):
    (a, b, loc_nx, scale_nx), dirac_proba_2 = s_x[self.cfx[x]]
    try:
        loc_nx = -x if loc_nx < -x else loc_nx
        scale_nx = self.cap if scale_nx > self.cap else scale_nx
        oh = max(abs(loc_nx), abs(scale_nx), self.m_tilde[x])
        # NOTE: the upper bound for s should be cap - x - l
        print((loc_nx, scale_nx, dirac_proba_2[0], dirac_proba_2[1]))
        nl, ns, p0, p1 = least_squares(find_intersections,
                               x0=(loc_nx, scale_nx, dirac_proba_2[0], dirac_proba_2[1]),
                               bounds=([-x, 0, 0, 0], [loc_nx, self.cap, 1, 1]),
                               args=(x, self.m_tilde[x], a, b, loc_nx, scale_nx, dirac_proba_2, oh, self.cap, False),
                               ftol=1e-3, method="dogbox").x
        if ns == 0:
            ns = scale_nx
        if j % p == 0:
            self.logger.info("     - l_hat and s_hat = {}, {} for m_hat(x) = {} => l_tilde and s_tilde = {}, {} "
                        "for m_tilde = {} < m_max = {}: {}% done".format("%.1f" % loc_nx, "%.1f" % scale_nx,
                                                                         "%.1f" % m_hat[x],
                                                                         "%.1f" % nl, "%.1f" % ns,
                                                                         "%.1f" % self.m_tilde[x], "%.1f" % self.m_max[x],
                                                                         (round(100 * j / len(dataset_sid[:-1]),
                                                                                3))))
    except Exception as e:
        print(e)
        if x != 0 and x != self.cap:  # bounds are equal for these cases
            nb_errors += 1
            # if self.m_tilde[x] > self.m_max[self.cfx[x]]:
            #     self.logger.error(
            #         "     * The MAE target {} is greater than the maximum target {}".format(round(self.m_tilde[x]),
            #                                                                                 round(self.m_max[self.cfx[x]])))
            self.logger.error(" * For x = {}, infeasible to meet the target exactly".format(x))
            self.logger.error(" {}".format(e))
        nl, ns = loc_nx, scale_nx
        p0, p1 = dirac_proba_2
    s_x_sid[x] = [a, b, nl, ns], [p0, p1]

# sim_params.cfx = sim_params.construct_cfx(smm.xyids[bus].dataset_x, smm.sids[bus].dataset_x)
# print(self.cfx)
# self.logger.info(loading_bar + "\nDetermination of the weight function om_tilde")
# sim_params.om_tilde, sim_params.e_score = sim_params.create_sid_weight_function(smm.xyids[bus].om, smm.xyids[bus].dataset_x, smm.sids[bus].dataset_x)

# from mape_maker.datasets.Dataset import load_spatial_data
#
# spatial_df_forecast = load_spatial_data("timeseries_data_files/WIND/DAY_AHEAD_wind.csv")
# spatial_df_actuals = load_spatial_data("timeseries_data_files/WIND/REAL_TIME_wind.csv")
# columns = list(set(list(spatial_df_forecast.columns)) & set(list(spatial_df_actuals.columns)))
# shared_index = list(set(list(spatial_df_forecast.datetime)) & set(list(spatial_df_actuals.datetime)))
# shared_index.sort()
#
# c = "150507_1_Wind"
# bus_node_str, _, energy_source = c.split("_")
# bus_node = int(bus_node_str)
# pre_df = pd.DataFrame(index=shared_index, columns=["datetime", "actuals", "forecasts"])
# pre_df["datetime"] = shared_index
# pre_df["actuals"] = spatial_df_actuals[c].loc[shared_index]
# pre_df["forecasts"] = spatial_df_forecast[c].loc[shared_index]
# #
# # # try:
# node_data = XYID(3, lat_long=smm.buses_to_lat_long[bus_node],
#                     base_process=None, start_date=kwargs["input_start_dt"],
#                     end_date=kwargs["input_end_dt"], ending_feature=kwargs["ending_feature"], xyid_load_pickle=False,
#                     logger=smm.logger, scale_by_capacity=False, df=pre_df, name=c)
#
#
# datasets[bus_node] = node_data
#
# key = 150507
# xyid = smm.xyids[key]
#
#
# xyid.s_x[xyid.dataset_x[792]]
#
# self = node_data
# x_bar, s_x_a = self.get_s_hat_datasetx_a()
#
# index_search = np.linspace(
#             0, self.dataset_info.get("cap"), XYID.len_s_hat)
# beta_parameters = [[0] * 4] * XYID.len_s_hat
# mean_var_sample = [[0] * 2] * XYID.len_s_hat
# dirac_parameters = [[0] * 2] * XYID.len_s_hat
# x_bar = np.array([0] * XYID.len_s_hat, dtype=float)
# a_past = 1
# b_past = 1
# for k, x_ in enumerate(index_search):
#     a_ = 3 if x_ == index_search[-1] else self.a
#     x_bar[int(k)], beta_parameters[k], mean_var_sample[k], dirac_parameters[k] = self.find_parameters(
#         x_, a_, a_past=a_past, b_past=b_past)
#     a_past, b_past = beta_parameters[k][:2]
#
# # s_a = node_data.estimate_parameters()
# s_x_a[list(s_x_a.keys())[-1]]
#
#
# self.s_x = dict([(key, []) for key in self.dataset_x])
# p = self.n_samples // 8
# for j, x in enumerate(self.dataset_x):
#     i = np.argmin(abs(x_bar - x))
#     nx = x_bar[i]
#     self.s_x[x] = copy.deepcopy(s_x_a[nx])
#     if x < nx:
#         self.s_x[x][0][2] = self.s_x[x][0][2] - (x-nx)
#     elif x > nx:
#         self.s_x[x][0][3] = self.s_x[x][0][3] - (x - nx)
#
#
# """For a given x_, find the sample with a% on the right and a% on the left, set its bounds (l and s) and
# find the shape parameters with the method of moment
#
# Args
#     x_: real around which to take the sample
#     a_: percent/2 of data for estimation samples
# """
# self = node_data
# x_ = self.dataset_x[-1]
# x_ = 77.65943877551021
# a_ = 3
# self.find_parameters(x_, a_)
#
# index_x_ = np.argwhere(self.dataset_x >= x_)[0][0]
# half_length_sample = int((a_ / 100) * self.n_different_samples)
# left_bound, right_bound = select_left_right_bound(index_x_, half_length_sample, self.n_different_samples)
# # print(left_bound, right_bound)
#
# interval_index = (self.x_t > self.dataset_x[left_bound]) & (
#     self.x_t < self.dataset_x[right_bound])  # strict condition used in v1
# x_bar = np.mean(self.x_t[interval_index])
# error_sample = self.e_t[interval_index]
# """
# Remove zero and cap non-zero measure
# """
# p_cap = self.y_t[interval_index][self.y_t[interval_index] >= self.cap].shape[0]/error_sample.shape[0]
# p_zero = self.y_t[interval_index][self.y_t[interval_index] <= 0.0].shape[0] / error_sample.shape[0]
#
# error_sample = error_sample.loc[(self.y_t[interval_index] > 0.0) & (self.y_t[interval_index] < self.cap)]
# error_sample = reject_outliers(error_sample)
# mean, var = np.mean(error_sample), np.std(error_sample) ** 2
# abs_lower = - x_bar
# abs_upper = self.dataset_info.get("cap") - x_bar
# if error_sample.shape[0] != 0:
#     if min(error_sample) < abs_lower:
#         lower = abs_lower
#     else:
#         lower = min(error_sample)
#     if max(error_sample) > abs_upper:
#         upper = self.dataset_info.get("cap") - x_bar - lower
#     else:
#         upper = max(error_sample) - lower
#     # lower = min(error_sample)
#     # upper = max(error_sample) - lower
#     if upper < 0:
#         print("PROBLEM ", x_)
#     # print("Problem in this fsolve")
#     #TODO add regularization with respect to past a, b
#     [a, b] = fsolve(find_alpha_beta, x0=np.array([a_past, b_past]), xtol=1e-2, args=(lower, upper, mean, var))
#     a = abs(a)
#     b = abs(b)
# else:
#     lower, upper = abs_lower, abs_upper
#     # print(p_zero, p_cap)
#     a, b = 1, 1


"""
fit_kernels estimate the marginal empirical variograms and fit 2 matern kernels w.r.t to space and time. Look at the plots!
"""
smm.fit_kernels(plot=True)

"""
Assuming a sum of the kernel to be the spatio-temporal kernel, the covariance matrix is easy to build. 
"""
smm.build_spatial_covariance()

"""
Simulate 5 actuals scenarios for December 31st.
"""
number_of_simulations = 10
results = smm.simulate(number_of_simulations=number_of_simulations, simulation_start_dt='2018-12-31 00:00:00', simulation_end_dt='2018-12-31 23:00:00')

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
