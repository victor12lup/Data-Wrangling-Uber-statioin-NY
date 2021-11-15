import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import reverse_geocode
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from pyproj import CRS
import seaborn as sns
import os
import mplleaflet
import descartes
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pandas.tseries.offsets import *
from datetime import datetime, timedelta
np.set_printoptions(precision=4, suppress=True)
import contextily as ctx
import osmnx as ox
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
import plotly
plotly.offline.init_notebook_mode(connected=True)
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.graph_objs as go
import gmplot



### READ IN DATA ###
def read_data():
    fileList = []
    filenames = []
    print("Reading in the Uber datasets as a dataframe...")
    for file in os.listdir():
        if file.startswith("uber-raw-data-"):
            filenames.append(file)
            df_month = pd.read_csv(file)
            fileList.append(df_month)
    #fileList.append(pd.read_csv("uber-raw-data-apr14.csv"))
    df = pd.concat(fileList)
    print("Sorting the dataframe...")
    df = df.sort_values(by=['Date/Time'])
    df['Datetime'] = pd.to_datetime(df['Date/Time'])
    df["Date"] = df["Datetime"]
    df.index = df["Datetime"]
    return df


def read_charging_station_data():
    print("Reading the charging station data...")
    url = "https://data.ny.gov/resource/7rrd-248n.json"
    df = pd.read_json(url)
    print(" \nCount total NaN (longitude, latitude): \n\n (",
          df.latitude.isnull().sum(), ",", df.longitude.isnull().sum(), ")")
    df_nyc = df
    return df_nyc


def make_geo_df(i_df, longitude_name, latitude_name):
    crs = {'init': 'epsg:4326'}
    geometry = [Point(xy) for xy in zip(i_df[longitude_name], i_df[latitude_name])]
    result_df = gpd.GeoDataFrame(i_df, crs=crs, geometry=geometry)
    return result_df


def remove_outside_nyc(df, longitude_name, latitude_name):
    G2 = ox.geocode_to_gdf('New York, New York, USA')
    geometry = [Point(xy) for xy in zip(df[longitude_name], df[latitude_name])]
    geom = G2.loc[0, 'geometry']
    inside = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        inside[i] = geom.intersects(geometry[i])
    df = df.loc[inside]
    return df


def add_boroughs(df, longitude_name, latitude_name):
    print("Adding boroughs...")
    nyc_boroughs = np.array(["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"])
    df["Borough"] = np.empty(len(df))
    for borough in range(0, len(nyc_boroughs)):
        location =  '{}, New York, New York, USA'.format(nyc_boroughs[borough])
        G2 = ox.geocode_to_gdf(location)
        geometry = [Point(xy) for xy in zip(df[longitude_name], df[latitude_name])]
        geom = G2.loc[0, 'geometry']
        inside = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            inside[i] = geom.intersects(geometry[i])
        df["Borough"].loc[inside] = nyc_boroughs[borough]
    return df


def plot_street_map(geo_df, geo_df_stations):
    G = ox.graph_from_place('New York, USA', network_type='drive')
    #Decomment below

    fig, ax = ox.plot_graph(G, figsize=(50, 20), bgcolor='black', node_color='w', node_size=0.3, edge_linewidth=0.3,
                            show=False, close=False)
    scatter = geo_df.plot(ax=ax, markersize=30, marker='.', color="red", alpha=1, zorder=7, label="Pick-up locations")
    scatter = geo_df_stations.plot(ax=ax, markersize=30, marker='o', color="lightskyblue", alpha=1, zorder=8, label="Charging stations")
    name = 'street_map_pickups_{}.png'.format(len(geo_df_stations))
    #"P"
    #legend = plt.legend(*scatter.legend_elements(num=5), loc="upper left", prop={'size': 13}, labelcolor='white', frameon=False)
    plt.legend(loc='upper left', prop={'size': 35}, labelcolor='white', markerscale=4)
    plt.set_facecolor('black')
    plt.savefig(name)
    plt.show()
    return G


def init_df_distances(len_df):
    id_closest_station = np.empty(len_df)
    dist_closest_station = np.empty(len_df)
    id_closest_station[:] = np.NaN
    dist_closest_station[:] = np.NaN
    d = {'ID_closest_station': id_closest_station, 'Distance': dist_closest_station}
    df_closest_distances = pd.DataFrame(data=d)
    return df_closest_distances


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def manhattan(lon1, lat1, lon2, lat2):
    diff_long = abs(lon1-lon2)
    diff_lat = abs(lat1-lat2)
    return diff_long + diff_lat


def nearest_polygon(df, name_longitude, name_latitude, G):
    lonPol = np.array(df[name_longitude].values)
    latPol = np.array(df[name_latitude].values)
    df["Polygon"] = ox.get_nearest_nodes(G, lonPol, latPol, 'balltree')
    return df


def calculate_shortest_distance(G, df, df_charging):
    print("Calculating the shortest path distances...")
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_nodes.head()
    # impute missing edge speeds then calculate edge travel times
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    # set to list because the original data type is not supported by ox
    lat_charging_list = df_charging["latitude"].tolist()
    long_charging_list = df_charging["longitude"].tolist()
    df_distances = init_df_distances(len(df))
    poligon_list = df_charging["Polygon"].tolist()
    number_of_nearest_stations = 10
    print(datetime.now().time())
    for pickup_loc in range(0, len(df)):
        dist_closest_charging_location = 999999
        charging_loc_index = 0
        circle_distance = np.empty(len(df_charging))
        circle_distance[:] = np.NaN
        d = {'Circle_distance': circle_distance, 'Charging_lon': long_charging_list, 'Charging_lat': lat_charging_list, 'Polygon': poligon_list}
        df_circle_distances = pd.DataFrame(data=d)
        for charging_loc in range(0, len(df_charging)):
            haversine_distance = manhattan(df["Lon"][pickup_loc], df["Lat"][pickup_loc], long_charging_list[charging_loc], lat_charging_list[charging_loc])
            df_circle_distances['Circle_distance'][charging_loc] = haversine_distance
        df_circle_distances = df_circle_distances.sort_values(by=['Circle_distance'])
        df_circle_distances = df_circle_distances[0:number_of_nearest_stations]
        try:
            for charging_loc in range(0, number_of_nearest_stations):
                route = ox.shortest_path(G, df["Polygon"].iloc[pickup_loc], df_circle_distances["Polygon"].iloc[charging_loc], weight='travel_time')
                # how long is our route in meters?
                edge_lengths = ox.utils_graph.get_route_edge_attributes(G, route, 'length')
                min_distance_meter = sum(edge_lengths)
                if min_distance_meter < dist_closest_charging_location:
                    dist_closest_charging_location = min_distance_meter
                    charging_loc_index = charging_loc
            df_distances["ID_closest_station"][pickup_loc] = df_circle_distances.index.values[charging_loc_index]
            df_distances["Distance"][pickup_loc] = dist_closest_charging_location
            print("The distance from pick-up location",  pickup_loc, "to the nearest charging location", df_circle_distances.index.values[charging_loc_index], "is", dist_closest_charging_location)
        except:
            df_distances["ID_closest_station"][pickup_loc] = np.nan
            df_distances["Distance"][pickup_loc] = np.nan
    print(datetime.now().time())
    print(df_distances)
    return df_distances
    # how long is our route in meters?

    #ox.plot_graph_route(G, route, node_size=0, route_linewidth=3)


def add_coordinates(df_distances, df):
    charging_lon = np.empty(len(df_distances))
    charging_lat = np.empty(len(df_distances))
    charging_lon[:] = np.NaN
    charging_lat[:] = np.NaN
    df_distances["Charging_lon"] = charging_lon
    df_distances["Charging_lat"] = charging_lat
    for i in range(len(df_distances)):
        try:
            location_charging_station = df_distances.iloc[i]["ID_closest_station"].astype(int)
            df_distances["Charging_lon"].iloc[i] = df_stations["geometry"].values.x[location_charging_station]
            df_distances["Charging_lat"].iloc[i] = df_stations["geometry"].values.y[location_charging_station]
        except IndexError:
            placeHolder = True
    df_distances["Pick_up_lon"] = df["Lon"].values
    df_distances["Pick_up_lat"] = df["Lat"].values
    return df_distances


def add_boroughs(df, longitude_name, latitude_name):
    nyc_boroughs = np.array(["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"])
    df["Borough"] = np.empty(len(df))
    for borough in range(0, len(nyc_boroughs)):
        location = '{}, New York, New York, USA'.format(nyc_boroughs[borough])
        print(location)
        G2 = ox.geocode_to_gdf(location)
        geometry = [Point(xy) for xy in zip(df[longitude_name], df[latitude_name])]
        geom = G2.loc[0, 'geometry']
        inside = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            inside[i] = geom.intersects(geometry[i])
        #df.loc[inside, df["Borough"]] = nyc_boroughs[borough]
        df["Borough"].loc[inside] = nyc_boroughs[borough]
         #= df.loc[inside]
    return df


def visualize_distances_map (df_distance, location_type):
    G = ox.graph_from_place('New York, USA', network_type='drive')
    fig, ax = ox.plot_graph(G, figsize=(50, 20), bgcolor='black', node_color='w', node_size=0.3, edge_linewidth=0.3,
                            show=False, close=False)
    plt.savefig("Only_street_map.pdf")
    plt.show()
    if location_type == 'pick-up':
        scatter = ax.scatter(df_distance["Pick_up_lon"], df_distance["Pick_up_lat"], c=df_distance["Distance"], cmap=plt.cm.RdYlGn_r, label='_nolegend_')
        legend1 = plt.legend(*scatter.legend_elements(num=5), loc="upper left", title="Distance in meters",
                                 prop={'size': 25}, markerscale = 4, labelcolor='white', frameon=False)
        title = legend1.get_title()
        title.set_fontsize(25)
        plt.setp(title, color='white')
        scatter2 = ax.scatter(df_distance["Charging_lon"], df_distance["Charging_lat"], color='deepskyblue', marker='x', s = 25, label="Charging station")

        plt.legend(loc='upper left',  prop={'size': 25}, labelcolor='white', markerscale=4)
        ax.add_artist(legend1)
    elif location_type == 'station':
        df_distance["Count"] = 1
        counts_distance = df_distance[['Charging_lon', 'Charging_lat', 'Count']]
        counts_distance = counts_distance.pivot_table(index=['Charging_lon', 'Charging_lat'], values=['Count'],
                                                      aggfunc=sum).sort_values('Count', ascending=False).reset_index()
        scatter = ax.scatter(counts_distance["Charging_lon"], counts_distance["Charging_lat"], c=counts_distance["Count"], s=25, cmap=plt.cm.Reds)
        legend3 = plt.legend(*scatter.legend_elements(num=5), loc="upper left", title="Charging location utility", prop={'size': 25}, markerscale=4, labelcolor='white', frameon=False)
        title = legend3.get_title()
        title.set_fontsize(25)
        plt.setp(title, color='white')
        ax.add_artist(legend3)
    plt.legend(loc='upper left', bbox_to_anchor=(-0.01, 0.8), prop={'size': 25}, markerscale=4, labelcolor='white', frameon=False)
    name = 'distance_map_{}.pdf'.format(location_type)
    plt.savefig(name)
    plt.show()


def aggregate_and_plot(df, to_aggregate, time_window):
    title = "Uber Arrivals in April to September, 2014"
    ylab_name = "Number of arrivals per " + to_aggregate
    if to_aggregate == "hour":
        agg_param = "H"
        title = "Hourly " + title
    elif to_aggregate == "day":
        agg_param = "D"
        title = "Daily " + title
    elif to_aggregate == "minute":
        agg_param = "T"
        title = "Minutely " + title
    elif to_aggregate == "30minutes":
        agg_param = "30T"
        title = title + " per 30 minutes"

    print("Aggregating the dataframe per", to_aggregate, "...")
    df = df.resample(agg_param).sum()
    smoothed_counts = df['Counts'].rolling(window=time_window).mean()
    df["Smoothed_Counts"] = smoothed_counts
    fig = df[["Counts", "Smoothed_Counts"]].plot()
    fig.set_title(title, fontsize = 15)
    fig.set_xlabel("Date", fontsize = 15)
    fig.set_ylabel(ylab_name, fontsize = 15)
    fig.legend(["Arrivals", "Smoothed arrivals"], markerscale = 12)
    plt.show()
    return df

# DETREND
def detrend_timeseries(df):
    diff = list()
    df_values = df["Counts"]
    diff.append(np.nan)
    for i in range(1, len(df)):
        value = df_values[i] - df_values[i - 1]
        diff.append(value)
    df["Diff"] = diff
    fig = df[["Diff"]].plot()
    fig.set_title("Detrended counts of Uber arrivals")
    fig.set_xlabel("Date")
    fig.set_ylabel("Relative number of arrivals")
    plt.show()
    return df


df = read_data()
df = add_boroughs(df, 'Lon', 'Lat')
df_stations = read_charging_station_data()
df["Counts"] = 1
"""
geo_df = make_geo_df(df, 'Lon', 'Lat')
geo_df_stations = make_geo_df(df_stations, 'longitude', 'latitude')
plot_street_map(geo_df, geo_df_stations)
print("ARRIVALS BEFORE REMOVAL", len(df))
"""
df = remove_outside_nyc(df, 'Lon', 'Lat')
print("ARRIVALS AFTER REMOVAL", len(df))
print("STATIONS BEFORE REMOVAL", len(df_stations))
df_stations = remove_outside_nyc(df_stations, 'longitude', 'latitude')
print("STATIONS AFTER REMOVAL", len(df_stations))

df_bronx = df.loc[df['Borough'] == "Bronx"]
print(df_bronx)
df_brooklyn = df.loc[df['Borough'] == "Brooklyn"]
df_manhattan = df.loc[df['Borough'] == "Manhattan"]
print(df_manhattan)
df_queens = df.loc[df['Borough'] == "Queens"]
df_staten_island = df.loc[df['Borough'] == "Staten Island"]

geo_df = make_geo_df(df, 'Lon', 'Lat')
geo_df_stations = make_geo_df(df_stations, 'longitude', 'latitude')
G = plot_street_map(geo_df, geo_df_stations)
df = nearest_polygon(df, 'Lon', 'Lat', G)
df_stations = nearest_polygon(df_stations, 'longitude', 'latitude', G)
df.to_csv("df_uber_polygon.csv", index=False)
df_stations.to_csv("df_stations_polygon.csv", index=False)
df_distances = calculate_shortest_distance(G, df, df_stations)
df_distances = add_coordinates(df_distances, df)
df_distances.dropna()
visualize_distances_map(df_distances, "pick-up")
visualize_distances_map(df_distances, "station")
df_minute = aggregate_and_plot(df, 'minute', 10080) #smoothed over a week
df_30minutes = aggregate_and_plot(df, '30minutes', 500)
df = aggregate_and_plot(df, 'hour', 168)  #smoothed over a week
df_day = aggregate_and_plot(df, 'day', 7)  #smoothed over a week
df = detrend_timeseries(df)
df_bronx = aggregate_and_plot(df_bronx,'hour', 168)
df_brooklyn = aggregate_and_plot(df_brooklyn,'hour', 168)
df_manhattan = aggregate_and_plot(df_manhattan, 'hour', 168)
df_queens = aggregate_and_plot(df_queens, 'hour', 168)
df_staten_island = aggregate_and_plot(df_staten_island, 'hour', 168)

avgHourPerWeekDay = df.groupby([df.index.dayofweek, df.index.hour])["Counts"].mean()
fig = avgHourPerWeekDay.plot()
labels = [item.get_text() for item in fig.get_xticklabels()]
list_weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for i in range(1,8):
    labels[i] = list_weekdays[i-1]
fig.set_xticklabels(labels, fontsize = 15)
plt.xticks(fontsize= 15)
fig.set_title("Average amount of Uber pick-ups per weekday", fontsize = 20)
fig.set_ylabel("Average number of pick ups", fontsize = 15)
fig.set_xlabel("Hour of the week", fontsize = 15)
fig.grid(linestyle='--')
plt.show()

avgHourPerWeekDayBronx = df_bronx.groupby([df_bronx.index.dayofweek, df_bronx.index.hour])["Counts"].mean()
avgHourPerWeekDayBrooklyn = df_brooklyn.groupby([df_brooklyn.index.dayofweek, df_brooklyn.index.hour])["Counts"].mean()
avgHourPerWeekDayQueens = df_queens.groupby([df_queens.index.dayofweek, df_queens.index.hour])["Counts"].mean()
avgHourPerWeekDayManhattan = df_manhattan.groupby([df_manhattan.index.dayofweek, df_manhattan.index.hour])["Counts"].mean()
avgHourPerWeekDayStatenIsland = df_staten_island.groupby([df_staten_island.index.dayofweek, df_staten_island.index.hour])["Counts"].mean()

fig = avgHourPerWeekDayManhattan.plot(color="green", label="Manhattan")
fig = avgHourPerWeekDayBrooklyn.plot(color="yellow", label="Brooklyn")
fig = avgHourPerWeekDayQueens.plot(color="orange", label="Queens")
fig = avgHourPerWeekDayBronx.plot(color="red", label="Bronx")
fig = avgHourPerWeekDayStatenIsland.plot(color="purple", label="Staten Island")
for i in range(1,8):
    labels[i] = list_weekdays[i-1]
plt.legend(loc='upper left', prop={'size': 15},  markerscale=12)
fig.set_xticklabels(labels, fontsize = 15)
plt.xticks(fontsize= 15)
fig.set_title("Average amount of Uber pick-ups per weekday, per borough", fontsize = 20)
fig.set_ylabel("Average number of pick ups", fontsize = 15)
fig.set_xlabel("Hour of the week", fontsize = 15)
fig.grid(linestyle='--')
plt.show()