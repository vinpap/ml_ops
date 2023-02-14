
"""Ce script est une modification du code présenté dans l'étape 4 pour optimiser 
les paramètres des modèles à l'aide de plusieurs techniques"""

import time
import mlflow
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn_genetic import GASearchCV
import sklearn_genetic
import skopt
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from memoized_property import memoized_property
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

# RMSE and training time (in seconds) for several optimization techniques
BEST_RMSE_WITH_GRIDSEARCH = 4.230879901223582
TRAINING_TIME_WITH_GRIDSEARCH = 267.96613907814026

BEST_RMSE_WITH_RANDOMIZED_SEARCH = 4.309053712469675
TRAINING_TIME_WITH_RANDOMIZED_SEARCH = 205.97612309455872

BEST_RMSE_WITH_BAYESIAN_SEARCH = 4.210947536613507
TRAINING_TIME_WITH_BAYESIAN_SEARCH = 440.6682639122009

BEST_RMSE_WITH_GA_SEARCH = 4.165002252559426 # For Genetic Algorithm Search
TRAINING_TIME_WITH_GA_SEARCH = 419.9572238922119



class Trainer:

	def __init__(self, model_name="linear", optimizer="grid_search"): 
		
		self.params = {}
		self.model_name = model_name
		self.optimizer = optimizer
		mlflow.set_experiment('TaxiFareModel')


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

		if self.model_name == "xgboost":

			self.model_params_ranges = {"xgboost__learning_rate": sp_randFloat(0.0001, 1), 
									"xgboost__max_depth": sp_randInt(1, 5),
									"xgboost__n_estimators": sp_randInt(10, 1000)}
			self.model_params_grid = {"xgboost__learning_rate": [0.001, 0.01, 0.1, 0.5, 1], 
									"xgboost__max_depth": [1, 2, 3, 4, 5],
									"xgboost__n_estimators": [10, 50, 100, 200, 500]}

			# Parameters spaces for Bayesian search
			self.model_params_spaces = {"xgboost__learning_rate": skopt.space.Real(1e-4, 1, prior='log-uniform'), 
									"xgboost__max_depth": skopt.space.Integer(1, 20),
									"xgboost__n_estimators": skopt.space.Integer(10, 1000)}

			# Parameters spaces for GA search
			self.model_params_spaces = {"xgboost__learning_rate": sklearn_genetic.space.Continuous(1e-4, 1, distribution='log-uniform'), 
									"xgboost__max_depth": sklearn_genetic.space.Integer(1, 10),
									"xgboost__n_estimators": sklearn_genetic.space.Integer(10, 1000)}

			pipe = Pipeline([
			('preproc', preproc_pipe),
			('xgboost', GradientBoostingRegressor())
			])
		else:
			pipe = Pipeline([
			('preproc', preproc_pipe),
			('linear', LinearRegression())
			])


		return pipe


	def evaluate(self, X_test, y_test, pipeline):
		'''returns a trained Search object'''
		rmse = pipeline.best_score_
		self.mlflow_log_metric("RMSE", rmse)
		self.params['TEST_SIZE'] = len(X_val)
		return (-pipeline.best_score_, pipeline.best_params_)

	def run(self, X_train, y_train, pipeline):
		'''returns a trained pipelined model'''


		rmse = make_scorer(compute_rmse, greater_is_better=False)
		match self.optimizer:
			case "grid_search":
				search = GridSearchCV(pipeline, 
									self.model_params_grid, 
									scoring=rmse, 
									n_jobs=2, 
									verbose=3, 
									refit=True)
			case "randomized_search": 
				search = RandomizedSearchCV(pipeline, 
											self.model_params_ranges, 
											scoring=rmse, 
											n_iter=50, 
											n_jobs=2, 
											verbose=3, 
											refit=True)

			case "bayes_search":
				search = skopt.BayesSearchCV(pipeline, 
											self.model_params_spaces, 
											scoring=rmse, 
											n_iter=50, 
											n_jobs=2, 
											verbose=3, 
											refit=True)

			case "genetic_search":
				search = sklearn_genetic.GASearchCV(pipeline, 
													param_grid=self.model_params_spaces, 
													scoring=rmse, 
													population_size=10, 
													generations=10, 
													cv=2, 
													n_jobs=-1,
													verbose=True, 
													refit=True)

			case _: raise ValueError("Unknown optimizer identifier")

		search.fit(X_train, y_train)
		pickle.dump(search.best_estimator_, open(f"best_{self.model_name}.pickle", "wb"))

		self.params['MODEL_NAME'] = self.model_name
		self.params['TRAIN_SIZE'] = len(X_train)
		self.mlflow_log_params(self.params)
		
		return search


def optimize_model(optimizer):
	"""This function creates a pipeline, trains it with the specified
	optimizer, saves the best model found and returns them."""

	print(f"Optimizing model with {optimizer}CV: ")
	model_trainer = Trainer(model_name="xgboost", optimizer=optimizer)
	t1 = time.time()
	model_trainer.mlflow_run()
	pipe = model_trainer.set_pipeline()
	pipe = model_trainer.run(X_train, y_train, pipe)
	rmse = model_trainer.evaluate(X_val, y_val, pipe)
	model_trainer.mlflow_end_run()
	t2 = time.time()
	
	print("Best parameters: ", rmse[1])
	print(f"Best RMSE: {rmse[0]}")
	print(f"Execution time: {t2-t1} s")
	print("Check out MLFlow dashboard on localhost for more info")



# store the data in a DataFrame
df = get_data()
df = clean_data(df)

# set X and y
y = df["fare_amount"]
X = df.drop("fare_amount", axis=1)
# hold out
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

#optimize_model("grid_search")
#optimize_model("randomized_search")
#optimize_model("bayes_search")
optimize_model("genetic_search")

