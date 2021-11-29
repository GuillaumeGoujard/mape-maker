import pandas as pd
import datetime as datetime
import logging
from mape_maker.datasets.Dataset import Dataset
from mape_maker.datasets.XYID import XYID
import skgstat as skg
import numpy as np

import pygeostat as gs

bus_list = pd.read_csv("Texas7k_20210529_BusData.csv")
data_buses = bus_list[['Substation Latitude', 'Substation Longitude', 'Number']].values
buses_to_lat_long = dict([(int(data_buses[i][2]), (data_buses[i][0], data_buses[i][1])) for i in range(data_buses.shape[0])])


spatial_df_forecast = pd.read_csv("timeseries_data_files/WIND/DAY_AHEAD_wind.csv")
spatial_df_forecast["datetime"] = spatial_df_forecast.apply(lambda row: datetime.datetime(int(row["Year"]), int(row["Month"]), int(row["Day"]), int(row["Period"])-1), axis=1)
spatial_df_forecast.index  = spatial_df_forecast["datetime"]

spatial_df_actuals = pd.read_csv("timeseries_data_files/WIND/REAL_TIME_wind.csv")
spatial_df_actuals["datetime"] = spatial_df_actuals.apply(lambda row: datetime.datetime(int(row["Year"]), int(row["Month"]), int(row["Day"]), int(row["Period"])-1), axis=1)
spatial_df_actuals.index = spatial_df_actuals["datetime"]

class NodeData:
    def __init__(self, name, lat_long, energy_source, bus_node, pre_df):
        self.name = name
        self.energy_source = energy_source
        self.lat_long = lat_long
        self.bus_node = bus_node
        start_date = pre_df.index[0].strftime("%Y-%m-%d %H:%M:%S")
        end_date = pre_df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        self.xyid = XYID(3, base_process=None, start_date=start_date,
                          end_date=end_date, ending_feature="actuals", xyid_load_pickle=True,
                          logger=logger, scale_by_capacity=None, df=pre_df, name=name)

class littleMM:
    def __init__(self):
        self.logger = logging.getLogger('mape-maker')

columns = list(set(list(spatial_df_forecast.columns)) & set(list(spatial_df_actuals.columns)))
shared_index = list(set(list(spatial_df_forecast.datetime)) & set(list(spatial_df_actuals.datetime)))
shared_index.sort()

start_date = shared_index[0].strftime("%Y-%m-%d %H:%M:%S")
end_date = shared_index[-1].strftime("%Y-%m-%d %H:%M:%S")

# for c in columns:
fit = True

if fit:
    datasets = {}
    logger = logging.getLogger('mape-maker')
    logger.level = logging.INFO
    c = columns[0]
    for k, c in enumerate(columns):
        if "_" in c:
            print(c, 100*k/len(columns))
            bus_node_str, _, energy_source = c.split("_")
            bus_node = int(bus_node_str)
            if bus_node in buses_to_lat_long:
                pre_df = pd.DataFrame(index=shared_index, columns=["datetime", "actuals", "forecasts"])
                pre_df["datetime"] = shared_index
                pre_df["actuals"] = spatial_df_actuals[c].loc[shared_index]
                pre_df["forecasts"] = spatial_df_forecast[c].loc[shared_index]

                # xyid = XYID(5, base_process=None, start_date=start_date,
                #             end_date=end_date, ending_feature="actuals", xyid_load_pickle=True,
                #             logger=logger, scale_by_capacity=None, df=pre_df, name=bus_node_str)
                try:
                    n = NodeData(c, buses_to_lat_long[bus_node], energy_source, bus_node, pre_df)
                    datasets[bus_node] = n
                except:
                    print("PROBLEM FOR ", c)
                    pass

    c = 240241
    import matplotlib.pyplot as plt

    longitudes = []
    latitudes = []
    for n in datasets:
        latitudes.append(datasets[n].lat_long[0])
        longitudes.append(datasets[n].lat_long[1])

    p = len(datasets)
    Distances = np.zeros((p,p))
    for i in range(p):
        lo, lat = longitudes[i], latitudes[i]
        for j in range(i+1, p):
            dist = np.sqrt((longitudes[j]-lo)**2 + (latitudes[j]-lat)**2)
            Distances[i,j] = Distances[j,i] = dist

    np.argmin(Distances+10*np.identity(p), axis=1)

    min_ = (Distances+10*np.identity(p)).min()
    np.argwhere(Distances+10*np.identity(p)==0.0)
    Distances[0, 1:].argmin()

    buses_in_datasets = list(datasets.keys())

    plt.scatter(x=np.array(longitudes)[[3, 32, 10]], y=np.array(latitudes)[[3, 32, 10]])
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.show()

    data = [datasets[buses_in_datasets[i]].xyid.arma_process.z_hat.values for i in range(len(buses_in_datasets))]
    spatial_base_process = pd.DataFrame(data=data).T
    spatial_base_process.columns = buses_in_datasets
    spatial_base_process.index = datasets[buses_in_datasets[0]].xyid.arma_process.z_hat.index


    k = 0
    def plot_baseprocess_index(k):
        x = np.array(longitudes)
        y = np.array(latitudes)
        z = spatial_base_process.iloc[k].values

        f, ax = plt.subplots(1,  sharex=True, sharey=True)
        tcf = ax.tricontourf(x,y,z, 20)  # np.linspace(-5, 5, 100) choose 20 contour levels, just to show how good its interpolation is
        ax.plot(x,y, 'ko ')
        f.colorbar(tcf)
        ax.set_title('Base Process RF at ' + str(spatial_base_process.index[k]))
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    # plt.colorbar(boundaries=np.linspace(0,1,5))
    # plt.show()
# self = littleMM()
# df = set_index(self, pre_df)

import imageio
import os

filenames = []
for k in range(200):
    # plot the line chart
    plot_baseprocess_index(k)

    # create file name and append it to a list
    filename = f'{k}.png'
    filenames.append(filename)
    # save frame
    plt.savefig(filename)
    plt.close()
# build gif
with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)

    import gstools as gs
    x = np.array(longitudes)
    y = np.array(latitudes)
    np.empty(0)
    final_emp = [np.zeros(16), np.zeros(16), np.zeros(16)]
    n = spatial_base_process.shape[0]
    for i in range(n):
        print(i)
        field = spatial_base_process.iloc[i].values
        emp_v = gs.vario_estimate((x, y), field, latlon=True, return_counts=True)
        final_emp[0] = emp_v[0]
        final_emp[1] += emp_v[1]
        final_emp[2] += emp_v[2]

    final_emp[1] = final_emp[1]/n


    t = np.array(range(spatial_base_process.shape[0]))
    t_bins = np.array(range(1, 20))
    final_emp_2 = [np.zeros(t_bins.shape[0]-1), np.zeros(t_bins.shape[0]-1), np.zeros(t_bins.shape[0]-1)]
    n = len(spatial_base_process.columns)
    for i in range(n):
        print(i)
        field = spatial_base_process[spatial_base_process.columns[i]].values
        emp_v = gs.vario_estimate(t, field, bin_edges=t_bins, latlon=False, return_counts=True, sampling_size=1000)
        final_emp_2[0] = emp_v[0]
        final_emp_2[1] += emp_v[1]
        final_emp_2[2] += emp_v[2]

    final_emp_2[1] = final_emp_2[1]/n


    sph = gs.Exponential(latlon=True, rescale=gs.EARTH_RADIUS)
    a, b = sph.fit_variogram(*final_emp)

    plt.scatter(final_emp[0], final_emp[1], label="empirical")
    xs_ = np.linspace(0, np.max(final_emp[0]), 100)
    plt.plot(xs_, [sph.variogram(x_) for x_ in xs_], color="red", label="Exponential Kernel")
    plt.title("Empirical Averaged Spatial Semi-variogram of Base Process")
    plt.xlabel("distance ")
    plt.ylabel("Variance")
    plt.legend()
    plt.show()

    sph2 = gs.Exponential()
    a, b = sph2.fit_variogram(*final_emp_2)

    plt.scatter(final_emp_2[0], final_emp_2[1])
    xs_ = np.linspace(0, np.max(final_emp_2[0]), 100)
    plt.plot(xs_, [sph2.variogram(x_) for x_ in xs_], color="red", label="Exponential Kernel")
    plt.title("Empirical Averaged Time Semi-variogram of Base Process")
    plt.xlabel("time (hours)")
    plt.ylabel("Variance")
    plt.show()


    from skgstat import SpaceTimeVariogram, Variogram

    # coord = np.array((x,y))
    # stv = Variogram(coord.T, spatial_base_process.values[0])
    # print(stv)
    # stv.plot()

    coord_time = [coord.T for _ in range(3)]
    stv = SpaceTimeVariogram(np.array(coord_time), spatial_base_process.values[:3], t_lags=2)

    xyt_field = np.empty((0,3))
    for t in range(3):
        array_xy = coord.T
        temp_xyt = np.concatenate([array_xy, t*np.ones((array_xy.shape[0],1))], axis=1)
        xyt_field = np.concatenate([xyt_field, temp_xyt], axis=0)


    field = spatial_base_process.values[:3].reshape(-1)

    emp_v = gs.vario_estimate((x, y), field, latlon=True, return_counts=True)
    emp_v = gs.vario_estimate(xyt_field.T, field, latlon=False, return_counts=True)


# import skgstat as skg

# from skgstat import data
# data = skg.data.pancake(N=500, seed=42)
# ax = sph.plot(x_max=2 * np.max(emp_v[0]))



# bins = np.arange(10)
# bin_center, gamma = gs.vario_estimate((x, y), field_1, bins)
# #
# fit_model = gs.Stable(dim=2)
# fit_model.fit_variogram(emp_v[0], final_emp/n, nugget=False)

# ax = fit_model.plot(x_max=8)
# ax.scatter(bin_center, gamma)
# print(fit_model)
#
# emp_v = gs.vario_estimate((x, y), field_1, latlon=True)
# sph = gs.Spherical(latlon=True, rescale=gs.EARTH_RADIUS)
# sph.fit_variogram(*emp_v, sill=np.var(field_1))
# # ax = sph.plot(x_max=2 * np.max(emp_v[0]))
#
# fig, ax = plt.subplots()
# sph.plot(x_max=2 * np.max(emp_v[0]), ax=ax)
# ax.scatter(*emp_v, label="Empirical variogram")
# ax.legend()
# print(sph)
# plt.show()
#
#
# # enclosing box for data points
# grid_lat = np.linspace(np.min(x[0]), np.max(x[0]))
# grid_lon = np.linspace(np.min(y[1]), np.max(y[1]))
# # ordinary kriging
# krige = gs.krige.Ordinary(sph, (x, y), field_1)
# krige((grid_lat, grid_lon), mesh_type="structured")
#
# fig, ax = plt.subplots()
# #
# # tcf = ax.tricontourf(x,y,z, 20)
# # plotting lat on y-axis and lon on x-axis
# ax.scatter(y, x, 50, c=field_1, edgecolors="k", label="input")
# krige.plot(ax=ax)
# ax.legend()
# plt.show()


import gstools as gs
import matplotlib.pyplot as plt

# # lat, lon, temperature
# data = np.array(
#     [
#         [52.9336, 8.237, 15.7],
#         [48.6159, 13.0506, 13.9],
#         [52.4853, 7.9126, 15.1],
#         [50.7446, 9.345, 17.0],
#         [52.9437, 12.8518, 21.9],
#         [53.8633, 8.1275, 11.9],
#         [47.8342, 10.8667, 11.4],
#         [51.0881, 12.9326, 17.2],
#         [48.406, 11.3117, 12.9],
#         [49.7273, 8.1164, 17.2],
#         [49.4691, 11.8546, 13.4],
#         [48.0197, 12.2925, 13.9],
#         [50.4237, 7.4202, 18.1],
#         [53.0316, 13.9908, 21.3],
#         [53.8412, 13.6846, 21.3],
#         [54.6792, 13.4343, 17.4],
#         [49.9694, 9.9114, 18.6],
#         [51.3745, 11.292, 20.2],
#         [47.8774, 11.3643, 12.7],
#         [50.5908, 12.7139, 15.8],
#     ]
# )
# pos = data.T[:2]  # lat, lon
# field = data.T[2]  # temperature
#
# emp_v = gs.vario_estimate(pos, field, latlon=True)
# sph = gs.Spherical(latlon=True, rescale=gs.EARTH_RADIUS)
# sph.fit_variogram(*emp_v, sill=np.var(field))
# fig, ax = plt.subplots()
# ax.scatter(*emp_v, label="Empirical variogram")
# sph.plot(x_max=2 * np.max(emp_v[0]), ax=ax)
# ax.legend()
# print(sph)
# plt.show()


"""
Beta check-up
"""
# c = columns[2]
# print(c, 100 * k / len(columns))
# bus_node_str, _, energy_source = c.split("_")
# bus_node = int(bus_node_str)
# if bus_node in buses_to_lat_long:
#     pre_df = pd.DataFrame(index=shared_index, columns=["datetime", "actuals", "forecasts"])
#     pre_df["datetime"] = shared_index
#     pre_df["actuals"] = spatial_df_actuals[c].loc[shared_index]
#     pre_df["forecasts"] = spatial_df_forecast[c].loc[shared_index]
#
#     xyid = XYID(5, base_process=None, start_date=start_date,
#                 end_date=end_date, ending_feature="actuals", xyid_load_pickle=False,
#                 logger=logger, scale_by_capacity=None, df=pre_df, name=bus_node_str)

# c = '240241_1_Wind',columns[2]
# logger = logging.getLogger('mape-maker')
# logger.level = logging.INFO
# k = 0
# print(c, 100*k/len(columns))
# bus_node_str, _, energy_source = c.split("_")
# bus_node = int(bus_node_str)
# if bus_node in buses_to_lat_long:
#     pre_df = pd.DataFrame(index=shared_index, columns=["datetime", "actuals", "forecasts"])
#     pre_df["datetime"] = shared_index
#     pre_df["actuals"] = spatial_df_actuals[c].loc[shared_index]
#     pre_df["forecasts"] = spatial_df_forecast[c].loc[shared_index]
# xyid = XYID(3, base_process=None, start_date=start_date,
#                           end_date=end_date, ending_feature="actuals", xyid_load_pickle=False,
#                           logger=logger, scale_by_capacity=None, df=pre_df, name="")
# #problem at 2018-01-07T01:00-02:00
# index_sd = 24*6+1 #xyid.x_t.index[24*6+1]
# p = xyid.s_x[xyid.x_t.iloc[index_sd]]
#
# j = 24*6+6
# p = xyid.s_x[xyid.x_t.iloc[j]]
# p, dirac_prob = xyid.s_x[xyid.x_t.iloc[j]]
#
# j = 24*1+15
# if xyid.y_t.iloc[j] <= 0:
#     y = dirac_prob[0]
# elif xyid.y_t.iloc[j] >= xyid.cap:
#     y = dirac_prob[1]
# else:
#     y = dirac_prob[0] + (1 - sum(dirac_prob)) * stats.beta.cdf(xyid.e_t.iloc[j], p[0], p[1], loc=p[2]-(xyid.x_t.iloc[j]-1.27), scale=p[3])
# y = 0.00001 if y == 0 else y
# y = 0.99999 if y == 1 else y
# z_hat[j] = norm.ppf(y)
#
# xyid.y_t[xyid.y_t >= xyid.cap]
#
# from scipy import stats
# p = xyid.s_x[xyid.x_t.iloc[index_sd]]
# y = stats.beta.cdf(
#     xyid.e_t.iloc[index_sd], p[0], p[1], loc=p[2], scale=p[3])
#
# p = xyid.s_x[xyid.x_t.iloc[index_sd+1]]
#
# y = stats.beta.cdf(
#     xyid.e_t.iloc[index_sd+1], p[0], p[1], loc=p[2], scale=p[3])
#
# stats.norm.ppf(y)
# #
# # p = xyid.s_x[0.0]
# # p[0] = 0.1
# # p[1] = 1
# # p = [2.31, 0.627, 0, 1]
# # x = np.linspace(stats.beta.ppf(0.01, p[0], p[1], loc=p[2], scale=p[3]),
# #                 stats.beta.ppf(0.99, p[0], p[1], loc=p[2], scale=p[3]), 200)
# #
# # fig, ax = plt.subplots(1, 1)
# # ax.plot(x, stats.beta.pdf(x, p[0], p[1], loc=p[2], scale=p[3]),
# #        'r-', lw=5, alpha=0.6, label='norm pdf')
# #
# # plt.show()
# #
# #
# x_ = xyid.x_t.iloc[index_sd+1]
#
# x_ = xyid.x_t.iloc[24*1+15]
# a_ = 5
# index_x_ = np.argwhere(xyid.dataset_x >= x_)[0][0]
# half_length_sample = int((a_ / 100) * xyid.n_different_samples)
# left_bound = index_x_ - half_length_sample if index_x_ - \
#     half_length_sample > 0 else 0
# right_bound = index_x_ + half_length_sample if index_x_ + half_length_sample < xyid.n_different_samples \
#     else xyid.n_different_samples - 1
#
# interval_index = (xyid.x_t > xyid.dataset_x[left_bound]) & (
#     xyid.x_t < xyid.dataset_x[right_bound])  # strict condition used in v1
# x_bar = np.mean(xyid.x_t[interval_index])
# error_sample = xyid.e_t[interval_index]
# mean, var = np.mean(error_sample), np.std(error_sample) ** 2
# abs_lower = - x_bar
# abs_upper = xyid.dataset_info.get("cap") - x_bar
#
# plt.plot(x, stats.beta.pdf(x, a, b, loc=lower, scale=upper),
#        'r-', lw=5, alpha=0.6, label='norm pdf')
#
# # plt.hist(error_sample)
# plt.show()
#
# error_sample_dirac = xyid.y_t[interval_index][xyid.y_t[interval_index]>0]
#
# error_sample = error_sample.loc[error_sample_dirac.index]
# stats.beta.fit(error_sample)
#
# lower = min(error_sample)
# upper = max(error_sample) - lower
# lower = -x_
# upper = xyid.dataset_info.get("cap") - x_bar - lower
# [a, b] = fsolve(find_alpha_beta, np.array([1, 1]), args=(lower, upper, mean, var))
#
# sum([stats.beta.pdf(e, a, b, loc=lower, scale=upper) for e in error_sample])
#
# a,b, lower, upper = [2.564671327602197, 0.5856283417205175, -28.618164903961933, 44.71739920844566]
#
# y = stats.beta.cdf(
#     xyid.e_t.iloc[index_sd+1], a, b, loc=lower, scale=upper)
#
# stats.norm.ppf(y)
#
# xyid.x_t[xyid.x_t>=xyid.dataset_x[-5]]