import pandas as pd
from Trips.utils import constants


class TrajectoryDF(pd.DataFrame):
    """TrajectoryDF.

    A TrajectoryDF object is a pandas.DataFrame that has three columns latitude, longitude and datetime.
    TrajDataFrame accepts the following keyword arguments:

    Parameters
    ----------
    data : list or dict or pandas DataFrame
        the data that must be embedded into a TrajDataFrame.

    latitude : int or str, optional
        the position or the name of the column in `data` containing the latitude. The default is `constants.LATITUDE`.

    longitude : int or str, optional
        the position or the name of the column in `data` containing the longitude. The default is `constants.LONGITUDE`.

    datetime : int or str, optional
        the position or the name of the column in `data` containing the datetime. The default is `constants.DATETIME`.

    user_id : int or str, optional
        the position or the name of the column in `data`containing the user identifier. The default is `constants.UID`.

    parameters : dict, optional
        parameters to add to the TrajDataFrame. The default is `{}` (no parameters).

    """

    def __init__(self, data, latitude=constants.LATITUDE, longitude=constants.LONGITUDE, datetime=constants.DATETIME,
                 user_id=constants.UID, speed=constants.SPEED, event=constants.EVENT, difference=constants.DIFFERENCE,
                 odometer=constants.ODOMETER, distance=constants.DISTANCE):

        original2default = {latitude: constants.LATITUDE,
                            longitude: constants.LONGITUDE,
                            datetime: constants.DATETIME,
                            user_id: constants.UID,
                            speed: constants.SPEED,
                            event: constants.EVENT,
                            difference: constants.DIFFERENCE,
                            odometer: constants.ODOMETER,
                            distance: constants.DISTANCE}

        columns = None
        if isinstance(data, pd.DataFrame):
            tdf = data.rename(columns=original2default)
            columns = tdf.columns
        else:
            raise TypeError('DataFrame constructor called with incompatible data and dtype: {e}'.format(e=type(data)))

        super(TrajectoryDF, self).__init__(tdf, columns=columns)
        self._set_traj(inplace=True)

    def _has_traj_columns(self):

        if (constants.DATETIME in self) and (constants.LATITUDE in self) and (constants.LONGITUDE in self):
            return True

        return False

    def _is_trajdataframe(self):

        if ((constants.DATETIME in self) and pd.core.dtypes.common.is_datetime64_any_dtype(self[constants.DATETIME]))\
                and ((constants.LONGITUDE in self) and pd.core.dtypes.common.is_float_dtype(self[constants.LONGITUDE])) \
                and ((constants.LATITUDE in self) and pd.core.dtypes.common.is_float_dtype(self[constants.LATITUDE])):

            return True

        return False

    def _set_traj(self, inplace=False):

        if not inplace:
            frame = self.copy()
        else:
            frame = self

        if not pd.core.dtypes.common.is_datetime64_any_dtype(frame[constants.DATETIME].dtype):
            frame[constants.DATETIME] = pd.to_datetime(frame[constants.DATETIME])

        if not pd.core.dtypes.common.is_float_dtype(frame[constants.LONGITUDE].dtype):
            frame[constants.LONGITUDE] = frame[constants.LONGITUDE].astype('float')

        if not pd.core.dtypes.common.is_float_dtype(frame[constants.LATITUDE].dtype):
            frame[constants.LATITUDE] = frame[constants.LATITUDE].astype('float')

        if not inplace:
            return frame

    def __getitem__(self, key):
        """
        If the result contains lat, lng and datetime, return a TrajDataFrame, else a pandas DataFrame.
        """
        result = super(TrajectoryDF, self).__getitem__(key)

        if (isinstance(result, TrajectoryDF)) and result._is_trajdataframe():
            result.__class__ = TrajectoryDF

        elif isinstance(result, TrajectoryDF) and not result._is_trajdataframe():
            result.__class__ = pd.DataFrame
        return result

    def settings_from(self, trajdataframe):
        """
        Copy the attributes from another TrajDataFrame.

        Parameters
        ----------
        trajdataframe : TrajDataFrame
            the TrajDataFrame from which to copy the attributes.
        """
        for k in trajdataframe.metadata:
            value = getattr(trajdataframe, k)
            setattr(self, k, value)

    @property
    def lat(self):
        if constants.LATITUDE not in self:
            raise AttributeError("The TrajDataFrame does not contain the column '%s.'" % constants.LATITUDE)
        return self[constants.LATITUDE]

    @property
    def lng(self):
        if constants.LONGITUDE not in self:
            raise AttributeError("The TrajDataFrame does not contain the column '%s.'" % constants.LONGITUDE)
        return self[constants.LONGITUDE]

    @property
    def datetime(self):
        if constants.DATETIME not in self:
            raise AttributeError("The TrajDataFrame does not contain the column '%s.'" % constants.DATETIME)
        return self[constants.DATETIME]

    @classmethod
    def sort_by_uid_and_datetime(self):
        if constants.UID in self.columns:
            return self.sort_values(by=[constants.UID, constants.DATETIME], ascending=[True, True])
        else:
            return self.sort_values(by=[constants.DATETIME], ascending=[True])
