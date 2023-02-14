from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data


class Trainer:

	def __init__(self): pass

	def set_pipeline(self):
	    '''returns a pipelined model'''
	    dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
	                     ('stdscaler', StandardScaler())])
	    time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
	                     ('ohe', OneHotEncoder(handle_unknown='ignore'))])
	    preproc_pipe = ColumnTransformer(transformers=[
	    ("distance", dist_pipe, ["pickup_latitude", 
	                                     "pickup_longitude", 
	                                     "dropoff_latitude",
	                                    "dropoff_longitude"]),
	    ("time", time_pipe, ["pickup_datetime"])
	    ])
	    pipe = Pipeline([
	    ('preproc', preproc_pipe),
	    ('linear_model', LinearRegression())
	    ])
	    return pipe


	def evaluate(self, X_test, y_test, pipeline):
	    '''returns the value of the RMSE'''
	    y_pred = pipe.predict(X_test)
	    rmse = compute_rmse(y_pred, y_test)
	    return rmse

	def run(self, X_train, y_train, pipeline):
	    '''returns a trained pipelined model'''
	    pipe.fit(X_train, y_train)
	    return pipeline

# store the data in a DataFrame
df = get_data()
df = clean_data(df)

# set X and y
y = df["fare_amount"]
X = df.drop("fare_amount", axis=1)

# hold out
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

model_trainer = Trainer()
pipe = model_trainer.set_pipeline()
pipe = model_trainer.run(X_train, y_train, pipe)
rmse = model_trainer.evaluate(X_val, y_val, pipe)
print(f"RMSE: {rmse}")