import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import skmob
from skmob.measures.individual import jump_lengths, radius_of_gyration, home_location, random_entropy, waiting_times, max_distance_from_home
from functools import reduce

if __name__ == "__main__":
    data = pd.read_pickle("/Users/sayehbayat/Documents/Data/stops.pkl")
    print(data.groupby(["uid"]).count())

    tdf = skmob.TrajDataFrame(data, latitude='TSLat', longitude='TSLong', datetime='TStime', user_id='uid')
    # compute the radius of gyration for each individual
    rg_df = radius_of_gyration(tdf)
    print(rg_df)

    jp_df = jump_lengths(tdf)
    jp_df['avg_jump_length'] = [np.mean(j) for j in jp_df.jump_lengths]
    print(jp_df)

    re_df = random_entropy(tdf)
    # jp_df['avg_jump_length'] = [np.mean(j) for j in jp_df.jump_lengths]
    print(re_df)

    df = reduce(lambda x, y: pd.merge(x, y, on='uid', how='outer'), [rg_df, wt_df, mxd_df, jp_df, re_df])
    df.to_pickle('/Users/sayehbayat/Documents/Data/spatial.pkl')

