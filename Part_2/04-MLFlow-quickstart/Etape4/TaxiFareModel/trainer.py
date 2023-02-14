import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from memoized_property import memoized_property
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

MLFLOW_URI = "https://mlflow.YOURSITE.co/"
EXPERIMENT_NAME = "[country code] [city] [login] model name + version" 

class Trainer:

	def __init__(self): 
		
		self.params = {}
		mlflow.set_experiment('TaxiFareModel')

	@memoized_property
	def mlflow_client(self):
		#mlflow.set_tracking_uri(MLFLOW_URI)
		return mlflow.MlflowClient()
	"""
	@memoized_property
	def mlflow_experiment_id(self):
		try:
			return self.mlflow_client.create_experiment(self.experiment_name)
		except BaseException:
			return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id"""

	def mlflow_run(self):
		mlflow.start_run(run_name = "Taxi Fares Model - Linear Regression")

	def mlflow_end_run(self):
		mlflow.end_run()

	def mlflow_log_params(self, params_dict):
		mlflow.log_params(params_dict)

	def mlflow_log_metric(self, key, value):
		mlflow.log_metric(key, value)


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
		self.mlflow_log_metric("RMSE", rmse)
		self.params['TEST_SIZE'] = len(X_val)
		return rmse

	def run(self, X_train, y_train, pipeline):
		'''returns a trained pipelined model'''
		pipe.fit(X_train, y_train)

		self.params['MODEL_NAME'] = "Linear Regression"
		self.params['TRAIN_SIZE'] = len(X_train)
		self.mlflow_log_params(self.params)
		
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

model_trainer.mlflow_run()

pipe = model_trainer.set_pipeline()
pipe = model_trainer.run(X_train, y_train, pipe)
rmse = model_trainer.evaluate(X_val, y_val, pipe)

model_trainer.mlflow_end_run()

print(f"RMSE: {rmse}")