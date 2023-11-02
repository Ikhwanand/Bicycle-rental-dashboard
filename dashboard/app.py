import streamlit as st 
import plotly.express as px 
import plotly.figure_factory as ff
import pandas as pd 
import json 
import requests
from streamlit_lottie import st_lottie
from numerize import numerize
import numpy as np
import plotly.graph_objects as go

# Utilise tout l'espace disponsible
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('.../style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_data():
    return pd.read_csv('day.csv', parse_dates=['dteday'])


def create_rfm_df(df):
    rfm = df.groupby('instant').agg({'dteday': lambda x: (pd.datetime.now() - x.max()).days,
                                       'casual': lambda x: len(x),
                                       'registered': lambda x: x.sum()}).reset_index()

    rfm.rename(columns={
        'dteday': 'recency',
        'casual':'frequency',
        'registered':'monetary'
    }, inplace=True)
    return rfm 

def create_daily_orders_df(df):
    """
    Create a dataframe with daily orders
    """
    daily_orders_df = df.resample('D', on='dteday').agg({
        'instant': 'nunique', 
        'cnt': 'sum'
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        'instant':'order_count',
        'cnt':'revenue'
    }, inplace=True)

    return daily_orders_df


def load_json():
    path = '../dashboard/Animation.json'
    with open(path, 'r') as file:
        url = json.load(file)
        return url 


df = load_data()
min_date = df['dteday'].min()
max_date = df['dteday'].max()

with st.sidebar:
    st_lottie(animation_data=load_json(), loop=True, quality='high', key='Bicycle')
    
    # Take start_date & end_date from date_input
    st.write("Choose your preferred date:")
    start_date, end_date = st.date_input(
        label='Time Span', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = df[(df['dteday'] >= str(start_date)) & (df['dteday'] <= str(end_date))]
main_df.sort_values(by='dteday', ascending=True, inplace=True)
main_df.reset_index(inplace=True, drop=True)

daily_order_df = create_daily_orders_df(main_df)
rfm_df = create_rfm_df(main_df)


st.header('Daily Count of Rental Bicycle Dashboard 🚲')
st.text("""
Data Set Information:
Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return back has become automatic. 
Through these systems, user is able to easily rent a bike from a particular position and return back at another position. 
Currently, there are about over 500 bike-sharing programs around the world which is composed of over 500 thousands bicycles. 
Today, there exists great interest in these systems due to their important role in traffic, environmental and health issues.
""")
st.subheader("Daily Orders")

met1, met2 = st.columns(2)

st.subheader('Best Customer Based on RFM Parameters')

col1, col2, col3 = st.columns(3)
with col1:
    avg_recency = numerize.numerize(rfm_df.recency.mean())
    st.metric('Average Recency (days)', value=avg_recency)

with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric('Average Frequency', value=avg_frequency)

with col3:
    avg_monetary = numerize.numerize(rfm_df.monetary.mean())
    st.metric('Average Monetary', value=avg_monetary)


left_tab1, left_tab2, right_tab1, right_tab2 = st.tabs(['Visual 1', 'Visual 2', 'Visual 3', 'Visual 4'])
with met1:
    total_orders = daily_order_df.order_count.sum()
    st.metric('Total Orders', value=total_orders)

with met2:
    total_revenue = numerize.numerize(daily_order_df.revenue.sum().astype('float'))
    st.metric('Total Revenue', value=total_revenue)


with left_tab1:
    st.subheader('Line Chart')
    fig = px.line(pd.melt(main_df, id_vars='dteday', value_vars=['casual', 'registered', 'cnt']).reset_index(), x='dteday', y='value', color='variable', title='Users Line plot')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

    
    fig_temp = px.line(pd.melt(main_df, id_vars='dteday', value_vars=['temp', 'atemp', 'hum', 'windspeed']).reset_index(), x='dteday', y='value', color='variable', title='Temperature Line plot')
    st.plotly_chart(fig_temp, use_container_width=True, config={'displayModeBar':True})


with left_tab2:
    st.subheader('Bar Plot')
    # Casual user plotting
    fig = px.histogram(main_df, x=['casual', 'registered', 'cnt'], title='Users Histogram')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':True})

    # temperature histogram
    fig_temp = px.histogram(main_df, x=['temp', 'atemp', 'hum', 'windspeed'], title='Temperature, Humidity and Windspeed Histogram')
    st.plotly_chart(fig_temp, use_container_width=True, config={'displayModeBar':True})

    season = {1:'Springer', 2:'Summer', 3:'Fall', 4:'Winter'}

    data_season = main_df.groupby('season')[['casual', 'registered', 'cnt']].sum().reset_index().assign(season=lambda x: x['season'].map(season))
    fig_season = px.bar(data_season, x='season', y=['casual', 'registered'], title='Casual vs Registered by Season')
    st.plotly_chart(fig_season, use_container_width=True, config={'displayModeBar':True})

    data_holiday = main_df.groupby('holiday')[['casual', 'registered']].sum().reset_index().assign(holiday=lambda x: x.holiday.map({1:'True', 0:'False'}))
    fig_holiday = px.bar(pd.melt(data_holiday, id_vars='holiday', value_vars=['casual', 'registered']), x='holiday', y='value', facet_col='variable', title='Bicycle users by Holiday')
    st.plotly_chart(fig_holiday, use_container_width=True, config={'displayModeBar':True})

    data_workingday = main_df.groupby('workingday')[['casual', 'registered']].sum().reset_index().assign(workingday=lambda x: x.workingday.map({1:'True', 0:'False'}))
    fig_workingday = px.bar(pd.melt(data_workingday, id_vars='workingday', value_vars=['casual', 'registered']), x='workingday', y='value', facet_col='variable', title='Bicycle users by Working Day')
    st.plotly_chart(fig_workingday, use_container_width=True)

    data_weekday=main_df.groupby('weekday')[['casual', 'registered', 'cnt']].sum().reset_index().assign(weekday=lambda x: x.weekday.map({0:'Sunday', 1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday'}))
    fig_weekday = px.bar(data_weekday, x='weekday', y=['casual', 'registered'], title='Bicycle users by Weekday')
    st.plotly_chart(fig_weekday, use_container_width=True, config={'displayModeBar':True})

    
    data_weather = main_df.groupby('weathersit')[['casual', 'registered']].sum().reset_index()
    fig_weather = px.bar(pd.melt(data_weather, id_vars='weathersit', value_vars=['casual', 'registered']), x='weathersit', y='value', facet_col='variable', title='Bicycle users by Weathersit')
    st.plotly_chart(fig_weather, use_container_width=True)
    st.text("""
    * weathersit:
        * 1: Clear, Few clouds, Partly cloudy, Partly cloudy
        * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
        * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
        * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
    """)

    
    fig_recency = px.bar(rfm_df.sort_values('recency', ascending=True).head(), x='instant', y='recency', title='By Recency (days)')
    st.plotly_chart(fig_recency, use_container_width=True, config={'displayModeBar':True})
    
    fig_frequency = px.bar(rfm_df.sort_values('frequency', ascending=False).head(5), x='instant', y='frequency', title='By Frequency')
    st.plotly_chart(fig_frequency, use_container_width=True)

    fig_monetary = px.bar(rfm_df.sort_values('monetary', ascending=False).head(5), x='instant', y='monetary', title='By Monetary')
    st.plotly_chart(fig_monetary, use_container_width=True, config={'displayModeBar':True})


with right_tab1:
    st.subheader('Scatter Plot')

    heatmap = go.Heatmap(z=main_df.values,
                     x=main_df.columns,
                     y=main_df.columns,
                     colorscale='Viridis')

    layout1 = go.Layout(title='Correlation Matrix')

    fig_corr = go.Figure(data=[heatmap], layout=layout1)
    st.plotly_chart(fig_corr, use_container_width=True)
    fig_season_2 = px.scatter(main_df, x='temp', y='atemp', facet_col='season', title='Temp vs Atemp')
    st.plotly_chart(fig_season_2, use_container_width=True)

    fig_temp_casual = px.scatter(main_df, x='temp', y='casual', facet_col='season', title='Casual Users vs Temp')
    st.plotly_chart(fig_temp_casual, use_container_width=True)
    
    fig_temp_regis = px.scatter(main_df, x='temp', y='registered', facet_col='season', title='Registered Users vs Temp')
    st.plotly_chart(fig_temp_regis, use_container_width=True)

    fig_weathersit2 = px.scatter(main_df, x='weathersit', y='hum', facet_col='season', title='Weathersit vs Hum')
    st.plotly_chart(fig_weathersit2, use_container_width=True)
    st.text("""
    * weathersit:
        * 1: Clear, Few clouds, Partly cloudy, Partly cloudy
        * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
        * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
        * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
    """)

    # clustering 
    np.random.seed(0)
    X = main_df[['casual', 'registered', 'cnt']].copy().values

    # set the number of clusters
    k = 3

    # Initialize centroids randomly
    centroids = X[np.random.choice(range(X.shape[0]), size=k, replace=False)]

    # Iterate until convergence
    for _ in range(10):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=1)
        labels = np.argmin(distances, axis=1)

        # update centroids based on the mean of teh assigned points
        for i in range(k):
            centroids[i] = np.mean(X[labels == i], axis=0)


    # plot the data points and centroids
    scatter = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=labels))
    centroid_scatter = go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker=dict(symbol='x', color='red'))

    layout = go.Layout(title='Clustering', xaxis=dict(title='X'), yaxis=dict(title='Y'))

    fig_cluster = go.Figure(data=[scatter, centroid_scatter], layout=layout)
    st.plotly_chart(fig_cluster, use_container_width=True)



with right_tab2:
    st.subheader('Box Plot')
    fig_users = px.box(main_df, y=['casual', 'registered', 'cnt'], title='Bicycle Users Box Plot')
    st.plotly_chart(fig_users, use_container_width=True)

    fig_condition = px.box(main_df, y=['temp', 'atemp', 'hum', 'windspeed'], facet_col='season', title='Temperature by Season')
    st.plotly_chart(fig_condition, use_container_width=True)

    box_weekday = px.box(main_df.assign(weekday=lambda x: x.weekday.map({0:'Sunday', 1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday'})), y=['cnt'], x='weekday', title='Bicycle user by weekday')
    st.plotly_chart(box_weekday, use_container_width=True)

    box_workingday = px.box(main_df, y='cnt', x='workingday', title='Bicycle user by working day')
    st.plotly_chart(box_workingday, use_container_width=True)

    box_season = px.box(main_df.assign(season=lambda x: x.season.map(season)), y=['cnt'], x='season', title='Bicycle user by season')
    st.plotly_chart(box_season, use_container_width=True)

    box_holiday = px.box(main_df, y=['cnt'], x='holiday', title='Bicycle user by holiday')
    st.plotly_chart(box_holiday, use_container_width=True)
