import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

def training_model():
    # 1. Load the dataset
    housing = pd.read_csv("housing.csv")

    # 2. Create a stratified test set
    housing['income_cat'] = pd.cut(housing["median_income"], 
                                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index].drop("income_cat", axis=1) # We will work on this data
        strat_test_set = housing.loc[test_index].drop("income_cat", axis=1) # Set aside the test data

    # We will work on the copy of training data 
    housing = strat_train_set.copy()

    # 3. Seperate features and labels
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)

    # print(housing, housing_labels)

    # 4. List the numerical and categorical columns
    num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    # 5. Lets make the pipeline 

    # For numerical columns
    num_pipline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
        # ("scaler", StandardScaler())
    ])

    # For categorical columns
    cat_pipline = Pipeline([ 
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipline, num_attribs), 
        ('cat', cat_pipline, cat_attribs)
    ])

    # 6. Transform the data
    housing_prepared = full_pipeline.fit_transform(housing)
    #print(housing_prepared.shape)

    # 7. Train the model
    xgboost = XGBRegressor()
    xgboost.fit(housing_prepared, housing_labels)
    
    with open("model.pkl","wb") as f:
        pickle.dump(xgboost,f)

    with open("pipeline.pkl","wb") as f:
        pickle.dump(full_pipeline,f)

    print("✅ model trained successfuly")

training_model()


# checking root_mean_squarred_error
# xgboost_preds = xgboost.predict(housing_prepared)
# xgboost_rmse = root_mean_squared_error(housing_labels, xgboost_preds)
# xgboost_rmses = -cross_val_score(xgboost, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(xgboost_rmses).describe())
