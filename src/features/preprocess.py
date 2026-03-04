import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

target = "Time_taken (min)"


def haversine(lat1, lon1, lat2, lon2):
    
    R = 6371
    
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def convert_mixed_time(value):

    if pd.isna(value):
        return np.nan

    value = str(value)

    
    if ":" in value:
        return pd.to_datetime(value, format="%H:%M", errors="coerce")

    
    try:
        float_val = float(value)
        hours = int(float_val * 24)
        minutes = int((float_val * 24 * 60) % 60)

        return pd.to_datetime(f"{hours}:{minutes}", format="%H:%M")

    except:
        return np.nan
    



def get_features_and_target(df):

    vehicle_condition_category = {0 :"Poor",1 :"Average", 2 :"Good",3:"Excellent"}

    df['Vehicle_condition'] = df['Vehicle_condition'].map(vehicle_condition_category)

    df = df[
        (df["Restaurant_latitude"].between(6, 37)) &
        (df["Delivery_location_latitude"].between(6, 37)) &
        (df["Restaurant_longitude"].between(68, 97)) &
        (df["Delivery_location_longitude"].between(68, 97))
    ]

    df = df[df["Delivery_person_Ratings"].between(1, 5)]



    df["distance_km"] = haversine(
                                df["Restaurant_latitude"],
                                df["Restaurant_longitude"],
                                df["Delivery_location_latitude"],
                                df["Delivery_location_longitude"]
                                )
    

    df["Time_Ordered"] = df["Time_Orderd"].apply(convert_mixed_time)

    df["order_hour"] = df["Time_Ordered"].dt.hour

    df["order_minute"] = df["Time_Ordered"].dt.minute

    df["Time_Order_picked"] = pd.to_datetime(
                                                df["Time_Order_picked"],
                                                format="%H:%M",
                                                errors="coerce"
                                            )

    df["pickup_delay_minutes"] = (df["Time_Order_picked"] - df["Time_Ordered"]).dt.total_seconds() / 60

    df["pickup_delay_minutes"] = df["pickup_delay_minutes"].fillna(df["pickup_delay_minutes"].median())

    df["Order_Date"] = pd.to_datetime(df["Order_Date"],format="%d-%m-%Y")

    df["order_day_of_week"] = df["Order_Date"].dt.dayofweek

    df["is_weekend"] = df["order_day_of_week"].isin([5,6]).astype(int)

    peak_hours  = (df.groupby("order_hour")["Time_taken (min)"].mean().sort_values(ascending=False).head(5).index)

    df["is_peak_hour"] = df["order_hour"].isin(peak_hours).astype(int)

    df.drop(columns=[
       "ID","Delivery_person_ID",
        "Restaurant_latitude","Restaurant_longitude",
        "Delivery_location_latitude","Delivery_location_longitude","Order_Date",
        "Time_Orderd","Time_Order_picked","Time_Ordered"],inplace=True)

    x = df.drop(target,axis=1)
    y = df[target]

    return x,y

def preprocessor_pipeline():

    categorical_features = [
        'Weather_conditions',
        'Vehicle_condition',
        'Road_traffic_density',
        'Type_of_order',
        'Type_of_vehicle',
        'Festival',
        'City'
    ]

    numerical_features = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "multiple_deliveries",
        "distance_km",
        "order_hour",
        "order_minute",
        "pickup_delay_minutes",
        "order_day_of_week",
        "is_weekend",
        "is_peak_hour"
    ]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])



    preprocessor = ColumnTransformer(

        transformers=[
            (
                "num",
                num_pipeline,
                numerical_features
            ),
            (
                "cat",
                cat_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor

def split_df(df):

    X,y = get_features_and_target(df)

    return train_test_split(X,y,test_size=0.3,random_state=56)