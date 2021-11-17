import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import operator

from datetime import datetime
import geopy.distance
import re
import os
import astral
from astral import LocationInfo
from astral.sun import sun
from itertools import groupby, tee, cycle


def format_time(data):
    data = data[data.Time != '.']
    data.reset_index(drop=True, inplace=True)
    times = []
    dates = []
    date_time = []
    for d, t in zip(data.Date, data.Time):
        date_i = datetime.strptime(d, '%d%b%Y')
        time_i = datetime.strptime(t, '%H:%M:%S')
        times.append(time_i)
        dates.append(date_i)
        date_time.append(datetime.combine(date_i, time_i.time()))

    data["Time"] = times
    data["Date_i"] = dates
    data["Date_Time"] = date_time

    return data


def check_time(A, l):
    diffA = (A[:-1]-A[1:])/1e9
    arr = diffA.astype('float64')
    i = np.where(arr < -40.0)[0]
    if len(i)>0:
        return l[:i[0]+1]
    else:
        return l


def extract_trip_features(dfi):
    L = [int(i != 0.0) for i in list(dfi.Vehicle_Speed)]
    idx = [[i for i, value in it] for key, it in itertools.groupby(enumerate(L),
                                                                   key=operator.itemgetter(1)) if key != 0.0]
    #dfi["Speed_Limit"].replace({".": np.nan}, inplace=True)
    dfi["Speed_Limit"] = dfi["Speed_Limit"].astype(float)
    #print(dfi.Difference)

    Vehicle_Speed = np.array(dfi.Vehicle_Speed)
    Latitude = np.array(dfi.Latitude)
    Longitude = np.array(dfi.Longitude)
    Speed_Limit = np.array(dfi.Speed_Limit)
    Event_Type = np.array(dfi.Event_Type)
    Time = np.array(dfi.Time)
    Date_Time = np.array(dfi.Date_Time)

    Speed_Diff = np.array(Vehicle_Speed - np.array(dfi.Speed_Limit))
    uid = dfi.uid.unique()

    trip_df = {"Vehicle_Speed": [], "Latitude": [], "Longitude": [], "Speed_Limit": [], "Event_Type": [],
               "Time": [], "Date_Time": [], "Speed_Diff": [], "DoW": [], "sunset": []}
    for l in idx:
        time_delta1 = pd.Timedelta(Time[l[0]] - Time[l[0] - 1]).total_seconds()
        if l[-1] + 1 < len(Time):
            time_delta2 = pd.Timedelta(Time[l[-1] + 1] - Time[l[-1]]).total_seconds()
            if 0 < time_delta1 < 120:
                l = [l[0] - 1] + l
            if 0 < time_delta2 < 120:
                l = l + [l[-1] + 1]
        else:
            if 0 < time_delta1 < 120:
                l = [l[0] - 1] + l
        A = Date_Time[l]
        l = check_time(A, l)
        trip_df["Vehicle_Speed"].append(Vehicle_Speed[l])
        trip_df["Latitude"].append(Latitude[l])
        trip_df["Longitude"].append(Longitude[l])
        trip_df["Speed_Limit"].append(Speed_Limit[l])
        trip_df["Speed_Diff"].append(Speed_Diff[l])
        trip_df["Event_Type"].append(Event_Type[l])
        trip_df["Time"].append(Time[l])
        trip_df["Date_Time"].append(Date_Time[l])

        city = LocationInfo(Latitude[l][0], Longitude[l][0])
        s = sun(city.observer, date=pd.to_datetime(Date_Time[l][0]).date()) #, tzinfo=city.timezone
        trip_df["sunset"].append(s["dusk"])

        #print(pd.to_datetime(Date_Time[l][0]).dayofweek)
        dow = pd.to_datetime(Date_Time[l][0]).dayofweek

        trip_df["DoW"].append(dow)
    trip_df["uid"] = [uid[0]] * len(trip_df["Date_Time"])
    return pd.DataFrame.from_dict(trip_df)


def create_trip_df(data):
    uid_list = data["uid"].unique()
    trip_list = []
    for uid in uid_list:
        dfi = data[data["uid"] == uid]
        trip_dfi = extract_trip_features(dfi[:])

        trip_dfi["len"] = [len(t) for t in trip_dfi.Vehicle_Speed]

        # Remove trips of < 3 min
        trip_dfi = trip_dfi[trip_dfi.len > 5]

        trip_dfi["Duration"] = [(d[-1] - d[0]) / np.timedelta64(60, 's')
                                for d in trip_dfi.Date_Time]

        trip_dfi["start_loc"] = [(lat[0], lon[0]) for lat, lon in
                                 zip(trip_dfi.Latitude, trip_dfi.Longitude)]

        trip_dfi["destination_loc"] = [(lat[-1], lon[-1]) for lat, lon in
                                       zip(trip_dfi.Latitude, trip_dfi.Longitude)]

        trip_list.append(trip_dfi)
    return pd.concat(trip_list)


def find_abs_acc(trip):
    """ Find Trip acceleration time series in m/s"""
    return np.abs(((trip[1:] - trip[:-1]) * 0.44703888888208) / 30)


def find_acc(trip):
    """ Find Trip acceleration time series in m/s"""
    return ((trip[1:] - trip[:-1]) * 0.44703888888208) / 30


def sequences(x):
    # find the boundaries where numbers are not consecutive
    boundaries = [i for i in range(1, len(x)) if x[i] > x[i-1]]
    # add the start and end boundaries
    boundaries = [0] + boundaries + [len(x)]
    # take the boundaries as pairwise slices
    slices = [boundaries[i:i + 2] for i in range(len(boundaries) - 1)]
    # extract all sequences with length greater than one
    return [x[start:end] for start, end in slices if end - start > 1]


def find_t_acc_dec(trip):
    """ Find Trip acceleration time series in m/s"""
    seq = list(sequences(trip))
    delta_t = [len(t)*0.5 for t in seq]
    return np.mean(delta_t)

def find_jerk(acc):
    """ Find Trip jerk time series in m/s2"""
    return np.abs((acc[1:] - acc[:-1]) / 30)


def find_over_speed(v_diff):
    """ Find Trip jerk time series in m/s2"""
    v = np.array(v_diff)
    v1 = v[v > 0]
    return sum(~np.isnan(v1))/sum(~np.isnan(v))


def find_under_speed(v_diff):
    """ Find Trip jerk time series in m/s2"""
    v = np.array(v_diff)
    v1 = v[v < 0]
    return sum(~np.isnan(v1))/sum(~np.isnan(v))

def find_season(d):
    # "day of year" ranges for the northern hemisphere
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    # winter = everything else

    if d in spring:
        season = 'spring'
    elif d in summer:
        season = 'summer'
    elif d in fall:
        season = 'fall'
    else:
        season = 'winter'
    return season

def add_features(tdf):

    tdf["acc"] = tdf["Vehicle_Speed"].apply(find_abs_acc)
    tdf["acc2"] = tdf["Vehicle_Speed"].apply(find_acc)
    tdf["jerk"] = tdf["acc2"].apply(find_jerk)
    tdf["t_acc"] = tdf["acc2"].apply(find_t_acc_dec)

    tdf["avg_v"] = tdf["Vehicle_Speed"].apply(np.mean)
    tdf["med_v"] = tdf["Vehicle_Speed"].apply(np.median)
    tdf["v_80"] = tdf["Vehicle_Speed"].apply(lambda x: np.percentile(x, 80))
    tdf["std_v"] = tdf["Vehicle_Speed"].apply(np.std)
    tdf["max_v"] = tdf["Vehicle_Speed"].apply(np.max)

    tdf["avg_acc"] = tdf["acc"].apply(np.mean)
    tdf["med_acc"] = tdf["acc"].apply(np.median)
    tdf["acc_80"] = tdf["acc"].apply(lambda x: np.percentile(x, 80))
    tdf["std_acc"] = tdf["acc"].apply(np.std)
    tdf["max_acc"] = tdf["acc"].apply(np.max)

    tdf["avg_jerk"] = tdf["jerk"].apply(np.mean)
    tdf["med_jerk"] = tdf["jerk"].apply(np.median)
    tdf["jerk_80"] = tdf["jerk"].apply(lambda x: np.percentile(x, 80))
    tdf["std_jerk"] = tdf["jerk"].apply(np.std)
    tdf["max_jerk"] = tdf["jerk"].apply(np.max)

    tdf["over_speed"] = tdf["Speed_Diff"].apply(find_over_speed)
    tdf["under_speed"] = tdf["Speed_Diff"].apply(find_under_speed)

    hard_break = []
    hard_cor_break = []
    sudden_acc = []

    for trip in tdf.acc2:
        #hard_break.append(sum(i < -1 for i in trip))
        #sudden_acc.append(sum(i > 1 for i in trip))
        hard_break.append(np.count_nonzero(trip == 'HARD_BREAKING_MESSAGE'))
        hard_cor_break.append(np.count_nonzero(trip == 'HARD_CORE_BRAKING_MESSAGE'))
        sudden_acc.append(np.count_nonzero(trip == 'HARD_ACCELERATION_MESSAGE'))

    tdf["hard_break"] = hard_break
    tdf["hard_cor_break"] = hard_cor_break
    tdf["sudden_acc"] = sudden_acc

    tdf["dist"] = [geopy.distance.distance(a, b).km
                   for a, b in zip(tdf.start_loc, tdf.destination_loc)]

    tdf['busday'] = [np.is_busday(np.datetime64(a[0], 'D')) for a in tdf.Date_Time]

    total_count = 0
    c_acc = []
    c_dec = []
    c_const = []
    for a in tdf.acc2:
        # total_count += len(a)
        c_acc.append(np.count_nonzero(a > 0) / len(a))
        c_dec.append(np.count_nonzero(a < 0) / len(a))
        c_const.append(np.count_nonzero(a == 0) / len(a))
    tdf["n_acc"] = c_acc
    tdf["n_dec"] = c_dec
    tdf["n_const"] = c_const

    tdf["late_night"] = [int(pd.to_datetime(a[-1]).time() > pd.to_datetime(b).time()) for a, b in
                         zip(tdf.Date_Time, tdf.sunset)]

    tdf['trip_date'] = [t[0] for t in tdf.Date_Time]
    tdf['trip_date'] = pd.to_datetime(tdf['trip_date'])

    tdf['busday'] = tdf['busday'].astype(int)
    tdf['late_night'] = tdf['late_night'].astype(int)

    w = []
    f = []
    su = []

    for td in tdf.trip_date:
        s = find_season(td.timetuple().tm_yday)
        if s == 'summer':
            su.append(1)
            f.append(0)
            w.append(0)
        elif s == 'winter':
            su.append(0)
            f.append(0)
            w.append(1)
        elif s == 'fall':
            su.append(0)
            f.append(1)
            w.append(0)
        else:

            su.append(0)
            f.append(0)
            w.append(0)
    tdf['winter'] = w
    tdf['summer'] = su
    tdf['fall'] = f
    return tdf

if __name__ == "__main__":
    os.chdir('..')

    data = pd.read_pickle("D:/breadcrumbs.pkl")
    print(len(data["uid"].unique()))
    print(data.columns)

    tdf = create_trip_df(data)
    tdf = add_features(tdf)
    tdf.to_pickle("D:/trips.pkl")

    print(tdf.shape)

