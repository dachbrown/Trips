
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
tqdm.pandas()
from Trips.utils import constants
from Trips.utils.utils import *
from astral import LocationInfo
from astral.sun import sun
from collections import Counter


def trip_metrics(tdf):

    # remove trips with no movements
    tdf['maxV'] = tdf[constants.SPEED].apply(np.max)
    tdf = tdf.loc[tdf['maxV'] > 0]

    # Trip start location and time
    tdf.loc[:, constants.ORIGIN_LAT], tdf.loc[:, constants.ORIGIN_LNG], tdf.loc[:, constants.ORIGIN_TIME] = \
        zip(*tdf.apply(lambda x: _origin(x[constants.LATITUDE], x[constants.LONGITUDE], x[constants.DATETIME]), axis=1))

    # Trip end location and time
    tdf.loc[:, constants.DESTINATION_LAT], tdf.loc[:, constants.DESTINATION_LNG], tdf.loc[:, constants.DESTINATION_TIME] = \
        zip(*tdf.apply(lambda x: _destination(x[constants.LATITUDE], x[constants.LONGITUDE], x[constants.DATETIME]), axis=1))

    # Find trip acceleration and jerk trajectory
    tdf.loc[:, constants.ACC] = [np.gradient(t, 1) for t in tdf[constants.SPEED]]
    tdf.loc[:, constants.JERK] = [np.gradient(t, 1) for t in tdf[constants.ACC]]

    # Find trip straightline and travelled distance
    tdf.loc[:, constants.TRAVELLED_DISTANCE] = tdf[constants.DISTANCE].apply(_travelled_dist)
    tdf.loc[:, constants.STRAIGHT_DISTANCE] = _straight_dist(zip(tdf[constants.ORIGIN_LAT],
                                                          tdf[constants.ORIGIN_LNG]),
                                                       zip(tdf[constants.DESTINATION_LAT],
                                                           tdf[constants.DESTINATION_LNG]))

    # speed, acceleration, and jerk statistics
    for v in [constants.SPEED, constants.ACC, constants.JERK]:
        print(v)
        tdf.loc[:, v+'_AVG'],  tdf.loc[:, v+'_MED'], tdf.loc[:, v+'_SD'], tdf.loc[:, v+'_80'], tdf.loc[:, v+'_MIN'], \
        tdf.loc[:, v+'_MAX'] = zip(*tdf[v].apply(_statistics))

    # Night trip or not
    tdf.loc[:, constants.NIGHT] = tdf.apply(lambda x: _dark(x[constants.LATITUDE], x[constants.LONGITUDE],
                                                     x[constants.DATETIME]), axis=1)
    # Number of special events
    tdf.loc[:, constants.HB], tdf.loc[:, constants.HCB], tdf.loc[:, constants.HA] = zip(*tdf[constants.EVENT].apply(_event_count))
    tdf.loc[:, constants.COUNT_OS],  tdf.loc[:, constants.DUR_OS] = zip(*tdf[constants.DIFFERENCE].apply(_over_speeding))
    tdf.loc[:, constants.COUNT_US], tdf.loc[:, constants.DUR_US] = zip(*tdf[constants.DIFFERENCE].apply(_under_speeding))

    # Trip duration
    tdf.loc[:, constants.DURATION] = tdf[constants.DATETIME].apply(_duration)

    # Trip hour
    tdf.loc[:, constants.HOUR] = tdf[constants.ORIGIN_TIME].apply(_hour)

    # Business day
    tdf.loc[:, constants.BUSINESS_DAY] = tdf[constants.ORIGIN_TIME].apply(_business_day)

    # Morning or Evening rush hour
    tdf.loc[:, constants.MORNING_RUSH] = tdf.apply(lambda x: _morn_rush(x[constants.HOUR],
                                                                        x[constants.BUSINESS_DAY]), axis=1)
    tdf.loc[:, constants.EVENING_RUSH] = tdf.apply(lambda x: _eve_rush(x[constants.HOUR],
                                                                       x[constants.BUSINESS_DAY]), axis=1)
    # distance group
    tdf.loc[:, constants.DIST_GRP] = _dist_groups(tdf[constants.TRAVELLED_DISTANCE])
    tdf = pd.get_dummies(tdf, columns=[constants.DIST_GRP])

    # day of week
    tdf.loc[:, constants.DOW] = tdf[constants.ORIGIN_TIME].apply(_day_of_week)
    
    return tdf


def _origin(lat, lng, tm):
    """
    Compute the start location and time of a trip
    :param lat: trip route latitude coordinates
    :param lng: trip route longitude coordinates
    :return: origin coordinates and trip start time
    """
    return lat[0], lng[0], tm[0]


def _destination(lat, lng, tm):
    """
    Compute the final destination and time of a trip
    :param lat: trip route latitude coordinates
    :param lng: trip route longitude coordinates
    :return: destination coordinates and trip end time
    """
    return lat[-1], lng[-1], tm[-1]


def _event_count(events):
    """
    Compute the number of HARD_BREAKING, HARD_CORE_BRAKING, HARD_ACCELERATION in a trip.
    :param events: list of event messages received from the car
    :return: number of HARD_BREAKING, HARD_CORE_BRAKING, HARD_ACCELERATION in a trip.
    """
    c = Counter(events)
    return c["HARD_BREAKING_MESSAGE"], c["HARD_CORE_BRAKING_MESSAGE"], c["HARD_ACCELERATION_MESSAGE"]


def _statistics(var):
    """
    Compute mean, median, standard deviation, 80th percentile, min, max for each variable
    :param var:
    :return: mean, median, standard deviation, 80th percentile, min, max
    """
    # Trim the leading and/or trailing zeros
    var = np.trim_zeros(var)
    return np.mean(var), np.median(var), np.std(var), np.percentile(var, 80), np.min(var), np.max(var)


def _dark(lat, lng, dt):
    """
    Find whether the trip was made after dusk ("dusk" is the period of twilight between complete darkness and sunrise)
    :param lat: destination latitude
    :param lng: destination longitude
    :param dt: date
    :return: 1 if the trip was made in dark and 0 otherwise
    """
    city = LocationInfo(lat[-1], lng[-1])
    d_time = pd.to_datetime(dt[-1])
    s = sun(city.observer, date=d_time.date()) #, tzinfo=city.timezone default UTC timezone
    return int(d_time.time() > pd.to_datetime(s['dusk']).time())


def _over_speeding(diff):
    """
    Compute count and duration of overspeeding (speed > 10mp + speed limit) events
    in a trip.
    :param diff: speed - speed_limit
    :return: count of overspeeding events, duration of overspeeding in seconds
    """
    cnt = np.sum(diff > 10)
    return cnt, cnt*30


def _under_speeding(diff):
    """
    Compute count and duration of underspeeding (speed > 10mp + speed limit) events
    in a trip.
    :param diff: speed - speed_limit
    :return: count of underspeeding events, duration of underspeeding in seconds
    """
    cnt = np.sum(diff < -10)
    return cnt, cnt*30


def _straight_dist(str_coord, end_coord):
    """
    Compute haversine distance from start location to the destination
    :param str_coord: start location lat, lng
    :param end_coord: destination lat, lng
    :return: straight line distance in miles
    """
    return [getDistance(a, b)*0.621371 for a, b in zip(str_coord, end_coord)]


def _travelled_dist(dist):
    return np.nanmax(dist)


def _day_of_week(dt):
    """
    Return the day of the week with Monday=0, Sunday=6.
    :param dt: start datetime of trip
    :return: day of the week
    """
    return pd.to_datetime(dt).weekday()


def _duration(dt):
    """
    Compute trip duration in seconds
    :param dt: time trajecotry
    :return: trip duration in seconds
    """
    return (dt[-1] - dt[0]) / np.timedelta64(60, 's')


def _dist_groups(travelled_dist):
    """
    :param travelled_dist: traveled distance in a trip by MI
    :return: trip distance category
    """
    bins = [0, 1, 5, 10, 20, 300000]
    category = ['Trips_Ends_1_Mile', 'Trips_Ends_5_Mile', 'Trips_Ends_10_Mile', 'Trips_Ends_20_Mile',
                'Trips_Ends_More_20_Mile']
    return pd.cut(travelled_dist, bins, labels=category)


def _business_day(dt):
    """
    Return whether trip was made during a business day
    :param dt: trip date
    :return: 1 if business day 0 otherwise
    """
    return int(np.is_busday(np.datetime64(dt, 'D'), weekmask=[1, 1, 1, 1, 1, 0, 0]))


def _morn_rush(hr, bd):
    """

    :param hr: trip hour
    :param bd: busness day (o/1)
    :return: 1 if the trip was made during the morning rush hour, 0 otherwise
    """
    return int(6 < hr < 10) * bd


def _eve_rush(hr, bd):
    """

    :param hr: trip hour
    :param bd: busness day (o/1)
    :return: 1 if the trip was made during the evening rush hour, 0 otherwise
    """
    return int(15 < hr < 19) * bd


def _hour(dt):
    return dt.hour


if __name__ == "__main__":
    data = pd.read_pickle("/Users/sayehbayat/Documents/Data/trips2.pkl")
    data = trip_metrics(data)
    print(np.percentile(data[constants.TRAVELLED_DISTANCE], 2.5))
    print(np.percentile(data[constants.TRAVELLED_DISTANCE], 97.5))
    print(np.percentile(data[constants.DURATION], 2.5))
    print(np.percentile(data[constants.DURATION], 97.5))
    print(data.head())
    data.to_pickle("/Users/sayehbayat/Documents/Data/trips_metrics.pkl")
