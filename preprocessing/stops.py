from Trips.utils import constants
from Trips.utils.utils import *


def stops(tdf):
    """
    Stop detection
    :param tdf:
    :return:
    """
    stp = tdf.groupby(tdf[constants.UID]).apply(lambda grp: _stops_individual(tdf))
    stp.reset_index(drop=True, inplace=True)
    return stp


def _stops_individual(tdf):
    """ Create a stop dataframe for a user

    :param tdf: trip dataframe
    :return: stop dataframe
    """
    sdf = tdf[[constants.DESTINATION_LAT, constants.DESTINATION_LNG, constants.DESTINATION_TIME]]
    sdf = sdf.rename(columns={constants.DESTINATION_TIME: constants.ORIGIN_TIME,
                              constants.DESTINATION_LAT: constants.ORIGIN_LAT,
                              constants.DESTINATION_LNG: constants.ORIGIN_LNG})

    sdf.loc[:, constants.DESTINATION_TIME] = list(tdf.loc[:, constants.ORIGIN_TIME])[1:] + \
                                             [tdf.loc[:, constants.ORIGIN_TIME].values[-1]]
    return sdf


if __name__ == "__main__":
    data = pd.read_pickle("D:/trips_metrics.pkl")
    SDF = stops(data)
    print(SDF)
    SDF.to_pickle("D:/stops.pkl")
    exit()
