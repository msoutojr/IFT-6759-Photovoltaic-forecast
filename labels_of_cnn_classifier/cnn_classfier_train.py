# -*- coding: utf-8 -*-
"""cnn_classfier_train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AgWKHPhcG7CYfdbmBLQJNIol_D9KReKX
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

sns.set(style="ticks")

"""test_classifier"""

datetime_trainval = pd.DataFrame(
    np.load(
        "/content/drive/MyDrive/asdf/photovoltaics/datetime_trainval.npy",
        allow_pickle=True,
    )[()],
    columns=["d_t"],
)

pv_log_trainval = pd.DataFrame(
    np.load(
        "/content/drive/MyDrive/asdf/photovoltaics/pv_log_trainval.npy",
        allow_pickle=True,
    )[()],
    columns=["pv_l"],
)

images_trainval = np.load(
    "/content/drive/MyDrive/asdf/photovoltaics/images_trainval.npy", allow_pickle=True
)[()]

list_zenith_test = []
list_azimuth_test = []

a = datetime_trainval["d_t"].dt.dayofyear
day_list = list(a)

latitude = 37.427764019872384

math.sin(math.radians(latitude))

list_azimuth_test = []
list_zenith_test = []

hour_series = datetime_trainval["d_t"].dt.hour
hour_list = list(hour_series)

len(day_list)
# alpha = (360/24)*((hour_list[i]-12))

for i in range(len(day_list)):
    lambd = 23.44 * math.sin(math.radians(360 * ((day_list[i] - 80) / 365.25)))
    hour_angle = (360 / 24) * (hour_list[i] - 12)
    # alpha = (360/24)*((hour_list[i]-12))
    zenith_angle = (
        math.acos(
            math.sin(math.radians(latitude)) * math.sin(math.radians(lambd))
            + math.cos(math.radians(latitude))
            * math.cos(math.radians(lambd))
            * math.cos(math.radians(hour_angle))
        )
        * 180
        / np.pi
    )
    azimth_angle = (
        math.atan(
            math.sin(math.radians(hour_angle))
            / (
                math.sin(math.radians(latitude)) * math.cos(math.radians(hour_angle))
                - math.cos(math.radians(latitude)) * math.tan(math.radians(lambd))
            )
        )
        * 180
        / np.pi
    )
    list_azimuth_test.append(azimth_angle)
    list_zenith_test.append(zenith_angle)

power_theta_list_test = []

for i in range(len(day_list)):
    power_theta = (
        1
        * 24.98
        * (
            math.cos(math.radians(22.5)) * math.cos(math.radians(list_zenith_test[i]))
            + math.sin(math.radians(22.5))
            * math.sin(math.radians(list_zenith_test[i]))
            * math.cos(math.radians(list_azimuth_test[i] - 195))
        )
    )
    power_theta_list_test.append(power_theta)

pv_log_trainval_list = list(pv_log_trainval["pv_l"])

len(pv_log_trainval_list)

plt.plot(pv_log_trainval_list)

plt.plot(power_theta_list_test)

r = [i / j for i, j in zip(power_theta_list_test, pv_log_trainval_list)]

for i in range(len(r)):
    if r[i] < 0.3:
        r[i] = "overcast"
    elif 0.3 <= r[i] < 0.5:
        r[i] = "cloudy-low"
    elif 0.5 <= r[i] < 0.7:
        r[i] = "cloudy-mid"
    elif 0.7 <= r[i] < 0.9:
        r[i] = "cloudy-high"
    elif 0.9 <= r[i] < 1.05:
        r[i] = "sunny"
    else:
        r[i] = "enhanced"

for i in range(len(r)):
    if r[i] < 0.3:
        r[i] = "overcast"
    elif 0.3 <= r[i] < 0.9:
        r[i] = "cloudy"
    elif 0.9 <= r[i] < 1.05:
        r[i] = "sunny"
    else:
        r[i] = "enhanced"

for i in range(len(r)):
    if r[i] < 0.3:
        r[i] = "overcast"
    elif 0.3 <= r[i] < 0.9:
        r[i] = "cloudy"
    else:
        r[i] = "sunny"

len(r)

plt.hist(r)

df_label = pd.DataFrame(r, columns=["label"])

df_label.to_csv("train_label_6_classes.csv")
