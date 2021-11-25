from Trips.utils import constants
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm
from Trips.core.trajectorydataframe import *
import inspect
tqdm.pandas()
from collections import Counter

def create_trip_df(tdf):
    """ Trip segmentation.

        Detect the trips for each individual in a TrajDataFrame. A trip is detected when the car sends a 'TRIP_START_MESSAGE' and ends when
        it sends a 'TRIP_END_MESSAGE'.

        Parameters
        ----------
        tdf : TrajDataFrame
            the input trajectories of the individuals.

        Returns
        -------
        TrajDataFrame
            a TrajDataFrame with the coordinates (latitude, longitude) of route.
    """
    uid_list = tdf[constants.UID].unique()
    trip_list = []
    for uid in uid_list:
        dfi = tdf[tdf[constants.UID] == uid]
        trip_list.append(_trip_segments(dfi))

    dfall = pd.concat(trip_list)
    dfall['Event_Start'] = [e[0] == 'TRIP_START_MESSAGE' for e in dfall[constants.EVENT]]
    dfall['Event_End'] = [e[-1] == 'TRIP_END_MESSAGE' for e in dfall[constants.EVENT]]
    return dfall[dfall.Event_Start & dfall.Event_End].copy()


def _trip_segments(tdf):

    # get indexes of all 'TRIP_START_MESSAGE', 'TRIP_END_MESSAGE'
    dfi = tdf.sort_values(by=[constants.UID, constants.DATETIME], ascending=[True, True])
    dfi.reset_index(inplace=True, drop=True)
    idx2 = _trips_sections(dfi[constants.EVENT])

    vehicle_speed = np.array(dfi[constants.SPEED])
    lat = np.array(dfi[constants.LATITUDE])
    lng = np.array(dfi[constants.LONGITUDE])
    speed_limit = np.array(dfi[constants.LIMIT])
    event_type = np.array(dfi[constants.EVENT])
    date_time = np.array(dfi[constants.DATETIME])
    difference = np.array(dfi[constants.DIFFERENCE])
    odometer = np.array(dfi[constants.ODOMETER])
    trip_distance = np.array(dfi[constants.DISTANCE])
    u = dfi[constants.UID].unique()[0]

    trip_df = {constants.SPEED: [], constants.LATITUDE: [], constants.LONGITUDE: [], constants.LIMIT: [], constants.EVENT: [],
               constants.DIFFERENCE: [], constants.DATETIME: [], constants.ODOMETER: [], constants.DISTANCE: [], constants.UID: []}

    for trip_idx in idx2:
        trip_df[constants.SPEED].append(vehicle_speed[trip_idx])
        trip_df[constants.LATITUDE].append(lat[trip_idx])
        trip_df[constants.LONGITUDE].append(lng[trip_idx])
        trip_df[constants.LIMIT].append(speed_limit[trip_idx])
        trip_df[constants.EVENT].append(event_type[trip_idx])
        trip_df[constants.DIFFERENCE].append(difference[trip_idx])
        trip_df[constants.DATETIME].append(date_time[trip_idx])
        trip_df[constants.ODOMETER].append(odometer[trip_idx])
        trip_df[constants.DISTANCE].append(trip_distance[trip_idx])
    trip_df[constants.UID] = [u] * len(idx2)
    return pd.DataFrame.from_dict(trip_df)


def _trips_sections(events):
    """
    Creates a trip when the car sends a 'TRIP_START_MESSAGE' and 'TRIP_END_MESSAGE'.
    """
    idx = [i for i, v in enumerate(events) if v in ['TRIP_START_MESSAGE', 'TRIP_END_MESSAGE']]
    return [list(range(idx[i], idx[i + 1] + 1)) for i in range(len(idx) - 1)]


if __name__ == "__main__":
    data = pd.read_pickle("D:/breadcrumbs.pkl")
    print(Counter(data.Event_Type))
    exit()
    print(data.columns)
    tdf = TrajectoryDF(data)
    print(type(tdf))
    print(tdf.columns)
    tdf = create_trip_df(tdf)
    tdf.to_pickle("D:/trips2.pkl")