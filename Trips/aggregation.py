import numpy as np
import pandas as pd
from pandas import Timestamp


def num_days(grp):
    grp = grp.sort_values(by="TStime").reset_index()
    l = list(grp.TStime)
    return (l[-1] - l[0]).days + 1


def monthly_var(df, var, p, mode='m'):
    if mode == 'm':
        df = df[['uid', p, var]].groupby([p, "uid"]).mean()
    else:
        df = df[['uid', p, var]].groupby([p, "uid"]).sum()
    df['day'] = df.index.get_level_values(0)
    df['uid'] = df.index.get_level_values(1)
    return df.reset_index(drop=True)

def user_measures(df):
    df['MornRush'] = [int(6 < v.hour < 10) for v in df.TStime]
    df['EveRush'] = [int(15 < v.hour < 19) for v in df.TStime]
    df['MornRush'] = df.MornRush & df.busday
    df['EveRush'] = df.EveRush & df.busday
    df['nTrips'] = 1

    A = df[['Duration', 'dist_meter', 'avg_v', 'med_v', 'v_80', 'std_v', 'max_v', 'avg_acc',
       'med_acc', 'acc_80', 'std_acc', 'max_acc', 'avg_jerk', 'med_jerk',
       'jerk_80', 'std_jerk', 'max_jerk', 'overspeed_perc', 'underspeed_perc', 'month_n',
            'NumHardBreak', 'NumHardCoreBreak', 'NumHardAcc', 'uid']].groupby(['month_n', "uid"]).mean()
    A.reset_index(level=['month_n', "uid"], inplace=True)
    print(A)
    B = df[['late_night', 'nTrips', 'dist_meter', 'MornRush', 'EveRush', 'speeding', 'over_speed', 'under_speed',
            'NumHardBreak', 'NumHardCoreBreak', 'NumHardAcc', 'month_n', 'uid']].groupby(['month_n', "uid"]).sum()
    B.reset_index(level=['month_n', "uid"], inplace=True)
    A = pd.merge(A, B, on=['uid', 'month_n'], how='left')

    B = df[['TStime', 'month_n', 'uid']].groupby(['month_n', "uid"]).apply(lambda grp: num_days(grp)).reset_index()
    B.set_axis(['month_n', 'uid', 'nDays'], axis='columns', inplace=True)
    print(B)
    A = pd.merge(A, B, on=['uid', 'month_n'], how='left')

    B = df[['TStime', 'month_n', 'uid']].groupby(['month_n', "uid"]).apply(lambda grp:
                                                                           len(grp["TStime"].dt.normalize().unique())).reset_index()

    B.set_axis(['month_n', 'uid', 'nDaysDriven'], axis='columns', inplace=True)
    print(B)
    A = pd.merge(A, B, on=['uid', 'month_n'], how='left')
    A['percNightTrips'] = A.late_night / A.nTrips
    A['percMornRush'] = A.MornRush / A.nTrips
    A['percEveRush'] = A.EveRush / A.nTrips
    A['percDaysDriven'] = A.nDaysDriven / A.nDays
    A['TripsperDay'] = A.nTrips / A.nDays
    A['SpeedingperTrip'] = A.speeding / A.nTrips
    return A


def time_selection(df, t1, t2):
    mask = (df['TStime'] >= t1) & (df['TStime'] < t2)
    return df.loc[mask][:]


if __name__ == "__main__":
    df = pd.read_pickle("D:/trips.pkl")
    t1 = Timestamp('2019-01-01 00:00:00')
    t2 = Timestamp('2020-01-01 00:00:00')
    df = time_selection(df, t1, t2)
    df['month_n'] = [d.month if d.year == 2019 else d.month + 12 for d in df.TStime]
    print(len(df["uid"].unique()))
    print(df.columns)
    print(df.month_n)

    print(np.percentile(df['Duration'], 2.5))
    print(np.percentile(df['Duration'], 97.5))
    df = df[df['Duration'] < np.percentile(df['Duration'], 97.5)]
    df = df[df['Duration'] > np.percentile(df['Duration'], 2.5)]
    print(np.percentile(df['dist_meter'], 2.5))
    print(np.percentile(df['dist_meter'], 97.5))
    df = df[df['dist_meter'] < np.percentile(df['dist_meter'], 97.5)]
    df = df[df['dist_meter'] > np.percentile(df['dist_meter'], 2.5)]

    df['speeding'] = df["Vehicle_Speed"].apply(lambda v: np.count_nonzero(v > 30))
    df = user_measures(df)

    df2 = pd.read_csv("D:/space_features.csv")
    print(df2)
    df = pd.merge(df, df2, left_on=['uid', 'month_n'], right_on=['uid', 'monthN'], how='left')
    df.to_csv("D:/users.csv")




