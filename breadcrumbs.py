import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def car_ids(df):
    drop_vid = ["Aaron's Car;", '2004Infiniti', "Bartek's;38.", 'Mary Kay;37.', "Bartek's;37.", 'Sarah Test f',
                'Vehicle Name', 'Mary Kay;38.', "Bartek's;36.", "Bartek's;35.", "Bartek's;34.", "Bartek's;33.",
                "Bartek's;32.", "Bartek's;31.", "Bartek's;30.", "Bartek's;29.", "Bartek's;28.", "Bartek's;27.",
                "Bartek's;26.", 'HighlanderTo', 'HyudaiElantr', "Bartek's;39.", '"','Ganesh B Tes', 'Study Vehicl', 'Vehicle_Name']
    df = df[~df.Vehicle_Name.isin(drop_vid)]; print(df.shape)
    print("Number of cars:", len(df.Vehicle_Name.unique()))
    replace_dict = {}
    for vid in df.Vehicle_Name.unique():
        if ';' in vid:
            replace_dict[vid] = vid[:-2]
    df.Vehicle_Name = df.Vehicle_Name.replace(replace_dict)
    return df


def add_uid(participants, df):
    result = participants[["deviceid", "id"]].groupby(["id"]).agg({'deviceid': list})
    result.reset_index(inplace=True)

    result.index = result.id
    result = result[["deviceid"]]
    map_id = result.to_dict()
    map_id = map_id['deviceid']

    pdf = []
    for k, v in map_id.items():
        dfi = []
        # print(k, v)
        for vid in v:
            dfi.append(df.loc[drive['Vehicle_Name'] == vid])
        dfi = pd.concat(dfi)
        dfi["uid"] = [k] * len(dfi)
        # if len(v)>2:
        # print(df.head)
        pdf.append(dfi)
    data = pd.concat(pdf)
    print(len(data.uid.unique()))
    return data


def format_time(data):
    data = data.dropna(subset=["Time", 'Date'])
    time_list = []
    date_list = []
    for d,t in zip(data.Date, data.Time):
        td = timedelta(seconds=t)
        time_list.append((d+td).strftime("%H:%M:%S"))
    return data


def format_time_2(data):
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


if __name__ == "__main__":
    drive = pd.read_sas("C:/Users/Sayeh/OneDrive - University of Toronto/Documents/Git/Drives/data/breadcrumbs_011121.sas7bdat",
                        encoding='latin1')
    drive = car_ids(drive)
    demo = pd.read_csv("C:/Users/Sayeh/OneDrive - University of Toronto/Documents/Git/Drives/data/merged_demo.csv")
    p = pd.read_csv(
        "C:/Users/Sayeh/OneDrive - University of Toronto/Documents/Git/Drives/data/participant_list_forCollab.csv")
    data = add_uid(p, drive)
    data = format_time(data)
    data = format_time_2(data)
    data.to_pickle("D:/breadcrumbs.pkl")
