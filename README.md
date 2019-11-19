
# Cross-Validation Training


```python
# Import necessary packages. 
import pandas as pd
import numpy as np 
import math
import time
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import copy
```

### Todo: 

* Examine the distribution of the predicted track and validation track. 
* Setting up the threshold for calculating the MSE specifically for peaks. 
* Creating Models for each histone marks.

## Data Preparation


```python
# Reading data.
ml_df = pd.read_csv("sources/ML_model/output/ml_data.csv", header=0)

# Create the shuffled dataframe for randomly selecting the folds. 
ml_df_shuf = ml_df.sample(frac=1)

# Define number of folds. 
k = 4

# Create index for the folds. 
folds_index = list(range(0, ml_df_shuf.shape[0], math.ceil(ml_df_shuf.shape[0]/k))) + [ml_df_shuf.shape[0]]
folds_index = [[folds_index[i]+1, folds_index[i+1]] for i in range(len(folds_index)-1)]
folds_index[0][0] = 0
folds_index
```




    [[0, 303615], [303616, 607230], [607231, 910845], [910846, 1214460]]




```python
ml_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avo</th>
      <th>cur</th>
      <th>cell</th>
      <th>mark</th>
      <th>ideas</th>
      <th>valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.246995</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.248117</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>92</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.248174</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.247927</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.248312</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.247233</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>19</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.247316</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>19</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.247123</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>43</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.246343</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>19</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.248208</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.248134</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.248359</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.248478</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.248123</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.248436</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.249592</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.248901</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.248855</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.250269</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.248659</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.248206</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>2</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.248728</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.249281</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.249216</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.248470</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.248940</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>19</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.249946</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.249419</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.249438</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>0</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.248860</td>
      <td>1.801744e-01</td>
      <td>C46</td>
      <td>M03</td>
      <td>15</td>
      <td>0.226440</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1214430</th>
      <td>0.182170</td>
      <td>2.794365e-01</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.134385</td>
    </tr>
    <tr>
      <th>1214431</th>
      <td>0.142648</td>
      <td>2.398810e-01</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214432</th>
      <td>0.089499</td>
      <td>7.356942e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214433</th>
      <td>0.085755</td>
      <td>7.330097e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214434</th>
      <td>0.088690</td>
      <td>5.963657e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214435</th>
      <td>0.074589</td>
      <td>5.904327e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214436</th>
      <td>0.071283</td>
      <td>5.898153e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214437</th>
      <td>0.068884</td>
      <td>5.900870e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214438</th>
      <td>0.062454</td>
      <td>6.090621e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214439</th>
      <td>0.066037</td>
      <td>5.905338e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214440</th>
      <td>0.081037</td>
      <td>5.905134e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214441</th>
      <td>0.079494</td>
      <td>5.905811e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214442</th>
      <td>0.079866</td>
      <td>5.901047e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214443</th>
      <td>0.080911</td>
      <td>5.904380e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214444</th>
      <td>0.078838</td>
      <td>5.904843e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214445</th>
      <td>0.064501</td>
      <td>5.938006e-02</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214446</th>
      <td>0.060386</td>
      <td>3.004827e-03</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214447</th>
      <td>0.058836</td>
      <td>6.579292e-04</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214448</th>
      <td>0.057562</td>
      <td>5.937061e-04</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214449</th>
      <td>0.058535</td>
      <td>1.503559e-03</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214450</th>
      <td>0.060157</td>
      <td>3.682951e-07</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214451</th>
      <td>0.062341</td>
      <td>0.000000e+00</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214452</th>
      <td>0.061212</td>
      <td>0.000000e+00</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214453</th>
      <td>0.061993</td>
      <td>0.000000e+00</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214454</th>
      <td>0.061545</td>
      <td>0.000000e+00</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214455</th>
      <td>0.063010</td>
      <td>0.000000e+00</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214456</th>
      <td>0.060978</td>
      <td>0.000000e+00</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214457</th>
      <td>0.061748</td>
      <td>0.000000e+00</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214458</th>
      <td>0.061063</td>
      <td>0.000000e+00</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1214459</th>
      <td>0.061999</td>
      <td>0.000000e+00</td>
      <td>C23</td>
      <td>M25</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1214460 rows × 6 columns</p>
</div>




```python
# Creating label and feature dataset. 
label = ml_df_shuf["valid"].to_numpy()
ml_df_shuf.drop(columns=["valid"])
features_dummies = pd.get_dummies(ml_df_shuf)
features = features_dummies.to_numpy() 
print(label[0:5])
print(features[0:5,:])
```

    [0.4194066  0.47912769 0.51961    0.499787   0.176605  ]
    [[0.468564   0.48123019 0.         0.4194066  0.         0.
      0.         0.         0.         0.         0.         1.
      0.         0.         0.         1.         0.        ]
     [0.51875847 0.31617134 0.         0.47912769 0.         0.
      0.         0.         0.         0.         1.         0.
      0.         0.         1.         0.         0.        ]
     [0.5656851  0.16588938 0.         0.51961    1.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         1.         0.         0.        ]
     [0.14765898 0.12652433 0.         0.499787   0.         0.
      0.         0.         0.         1.         0.         0.
      0.         0.         1.         0.         0.        ]
     [0.18135708 0.17125739 0.         0.176605   0.         0.
      0.         0.         0.         0.         0.         0.
      0.         1.         0.         0.         1.        ]]


## Cross-Validation Training


```python
def time_stamp(): 
    print("[{}]".format(time.time()))
```


```python
def cv_train(model_name, label, feature, folds): 
    predicted_set = {}
    iter_index = 0

    for validation_set in folds: 
        print("---- Creating Validation Set for data in range {}".format(validation_set))
        if model_name == "RF": 
            regr = RandomForestRegressor()
        else: 
            # Part for adding more models later. 
            print("#### Invalid Model Name. Terminating Training")
            break

        # Creating train and valid features, labels. 
        curr_test_feature = feature[validation_set[0]:validation_set[1],:]
        curr_test_label = label[validation_set[0]:validation_set[1]]
        curr_train_feature = np.delete(feature, np.s_[validation_set[0]:validation_set[1]+1], axis=0)
        curr_train_label = np.delete(label, np.s_[validation_set[0]:validation_set[1]+1], axis=0)

        start_time = time.time()
        print("-------- Start Training on {}".format(start_time))
        regr.fit(curr_train_feature, curr_train_label)
        print("-------- Finished Training, elapsed time: {}".format(time.time() - start_time))
        predicted_set[iter_index] = {
            "model": regr, 
            "predicted": regr.predict(curr_test_feature),
            "test_label": curr_test_label,
            "avocado": curr_test_feature[:,0],
            "curr_impute": curr_test_feature[:,1]
        }
        iter_index += 1

    return predicted_set
```


```python
def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))
```


```python
model_dic = cv_train("RF", label, features, folds_index)
```

    ---- Creating Validation Set for data in range [0, 303615]
    -------- Start Training on 1574182987.977485
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 58.52015209197998
    ---- Creating Validation Set for data in range [303616, 607230]
    -------- Start Training on 1574183048.340859
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 58.19038510322571
    ---- Creating Validation Set for data in range [607231, 910845]
    -------- Start Training on 1574183108.437985
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 57.9245810508728
    ---- Creating Validation Set for data in range [910846, 1214460]
    -------- Start Training on 1574183168.262582
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 57.9299750328064



```python
# Calculating the MSE for each fold. 
MSE_dict = {}
for i in range(4): 
    MSE_dict[i] = {
        "predicted": round(np.mean(abs(model_dic[i]["predicted"]-model_dic[i]["test_label"])), 4), 
        "avocado": round(np.mean(abs(model_dic[i]["avocado"]-model_dic[i]["test_label"])), 4),
        "curr_impute": round(np.mean(abs(model_dic[i]["curr_impute"]-model_dic[i]["test_label"])), 4)
    }
pretty(MSE_dict)
```

    0
    	predicted
    		0.1372
    	avocado
    		0.1537
    	curr_impute
    		0.2792
    1
    	predicted
    		0.1402
    	avocado
    		0.1577
    	curr_impute
    		0.2832
    2
    	predicted
    		0.1361
    	avocado
    		0.1529
    	curr_impute
    		0.277
    3
    	predicted
    		0.1367
    	avocado
    		0.1546
    	curr_impute
    		0.2823



```python
# Pairwise Data Visualization. 
for i in range(4): 
    temp_dic = copy.deepcopy(model_dic[i])
    temp_dic.pop("model")
    temp_plot_df = pd.DataFrame(temp_dic)
    temp_plot_df.plot(subplots=True, figsize=(10, 6))
```


![svg](output_13_0.svg)



![svg](output_13_1.svg)



![svg](output_13_2.svg)



![svg](output_13_3.svg)



```python
# Plot the feature importance. 
feat_imp_li = model_dic[0]["model"].feature_importances_
feat_importance = pd.Series(feat_imp_li, index=features_orig.columns)
feat_importance.nlargest(6).plot(kind='barh')
# print(model_dic[0]["model"].feature_importances_)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1abfdb58d0>




![svg](output_14_1.svg)


## Cross-Validated Training on Datasets Without IDEAS State


```python
print(features[0,:])
```

    [0.2037363  0.25777569 4.         0.         0.         1.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         1.        ]



```python
ml_df_shuf.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avo</th>
      <th>cur</th>
      <th>cell</th>
      <th>mark</th>
      <th>ideas</th>
      <th>valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>792090</th>
      <td>0.468564</td>
      <td>0.481230</td>
      <td>C32</td>
      <td>M21</td>
      <td>0</td>
      <td>0.419407</td>
    </tr>
    <tr>
      <th>1151646</th>
      <td>0.518758</td>
      <td>0.316171</td>
      <td>C31</td>
      <td>M03</td>
      <td>0</td>
      <td>0.479128</td>
    </tr>
    <tr>
      <th>384424</th>
      <td>0.565685</td>
      <td>0.165889</td>
      <td>C17</td>
      <td>M03</td>
      <td>0</td>
      <td>0.519610</td>
    </tr>
    <tr>
      <th>579290</th>
      <td>0.147659</td>
      <td>0.126524</td>
      <td>C27</td>
      <td>M03</td>
      <td>0</td>
      <td>0.499787</td>
    </tr>
    <tr>
      <th>956422</th>
      <td>0.181357</td>
      <td>0.171257</td>
      <td>C46</td>
      <td>M25</td>
      <td>0</td>
      <td>0.176605</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove all IDEAS State dummies variable. 
features_wt_idea = ml_df_shuf[["avo", "cur", "cell", "mark"]]
features_wt_idea = pd.get_dummies(features_wt_idea).to_numpy()
features_wt_idea[0,:]
```




    array([0.468564  , 0.48123019, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 1.        ,
           0.        , 0.        , 0.        , 1.        , 0.        ])




```python
model_dic_id_rmd = cv_train("RF", label, features_wt_idea, folds_index)
```

    ---- Creating Validation Set for data in range [0, 303615]
    -------- Start Training on 1574183999.332468
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 56.25135779380798
    ---- Creating Validation Set for data in range [303616, 607230]
    -------- Start Training on 1574184057.5759418
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 55.562814235687256
    ---- Creating Validation Set for data in range [607231, 910845]
    -------- Start Training on 1574184115.145239
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 63.64612078666687
    ---- Creating Validation Set for data in range [910846, 1214460]
    -------- Start Training on 1574184181.19211
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 67.51104378700256



```python
# Calculating the MSE for each fold. 
MSE_dict = {}
for i in range(4): 
    MSE_dict[i] = {
        "predicted": round(np.mean(abs(model_dic_id_rmd[i]["predicted"]-model_dic_id_rmd[i]["test_label"])), 4), 
        "avocado": round(np.mean(abs(model_dic_id_rmd[i]["avocado"]-model_dic_id_rmd[i]["test_label"])), 4),
        "curr_impute": round(np.mean(abs(model_dic_id_rmd[i]["curr_impute"]-model_dic_id_rmd[i]["test_label"])), 4)
    }
pretty(MSE_dict)
```

    0
    	predicted
    		0.1371
    	avocado
    		0.155
    	curr_impute
    		0.2808
    1
    	predicted
    		0.1385
    	avocado
    		0.1552
    	curr_impute
    		0.2796
    2
    	predicted
    		0.1389
    	avocado
    		0.1541
    	curr_impute
    		0.2796
    3
    	predicted
    		0.1381
    	avocado
    		0.1545
    	curr_impute
    		0.2817


## Creating Models for Each Histone Marks

### Data Preparation


```python
ml_df_shuf.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avo</th>
      <th>cur</th>
      <th>cell</th>
      <th>mark</th>
      <th>ideas</th>
      <th>valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>792090</th>
      <td>0.468564</td>
      <td>0.481230</td>
      <td>C32</td>
      <td>M21</td>
      <td>0</td>
      <td>0.419407</td>
    </tr>
    <tr>
      <th>1151646</th>
      <td>0.518758</td>
      <td>0.316171</td>
      <td>C31</td>
      <td>M03</td>
      <td>0</td>
      <td>0.479128</td>
    </tr>
    <tr>
      <th>384424</th>
      <td>0.565685</td>
      <td>0.165889</td>
      <td>C17</td>
      <td>M03</td>
      <td>0</td>
      <td>0.519610</td>
    </tr>
    <tr>
      <th>579290</th>
      <td>0.147659</td>
      <td>0.126524</td>
      <td>C27</td>
      <td>M03</td>
      <td>0</td>
      <td>0.499787</td>
    </tr>
    <tr>
      <th>956422</th>
      <td>0.181357</td>
      <td>0.171257</td>
      <td>C46</td>
      <td>M25</td>
      <td>0</td>
      <td>0.176605</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print unique marks. 
marks_list = ml_df_shuf.mark.unique()
print(marks_list)
```

    ['M21' 'M03' 'M25']



```python
def hist_models(mark_id, orig_df, ideas=True):
    df = orig_df[orig_df.mark == mark_id]
    label = df["valid"].to_numpy()
    if ideas: 
        features = pd.get_dummies(df[["avo", "cur", "cell", "ideas"]]).to_numpy()
    else: 
        features = pd.get_dummies(df[["avo", "cur", "cell"]]).to_numpy()

    # Define number of folds. 
    k = 4

    # Create index for the folds. 
    folds_index = list(range(0, df.shape[0], math.ceil(df.shape[0]/k))) + [df.shape[0]]
    folds_index = [[folds_index[i]+1, folds_index[i+1]] for i in range(len(folds_index)-1)]
    folds_index[0][0] = 0
    folds_index

    model_dic = cv_train("RF", label, features, folds_index)

    MSE_dict = {}
    for i in range(4): 
        MSE_dict[i] = {
            "predicted": round(np.mean(abs(model_dic[i]["predicted"]-model_dic[i]["test_label"])), 4), 
            "avocado": round(np.mean(abs(model_dic[i]["avocado"]-model_dic[i]["test_label"])), 4),
            "curr_impute": round(np.mean(abs(model_dic[i]["curr_impute"]-model_dic[i]["test_label"])), 4)
        }
    pretty(MSE_dict)

    return {
        "model_dic": model_dic, 
        "MSE": MSE_dict
    }
```

### Model for M21 - H3K4me2


```python
m21_result = hist_models("M21", ml_df_shuf)
```

    ---- Creating Validation Set for data in range [0, 116775]
    -------- Start Training on 1574188344.497136
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 17.838374853134155
    ---- Creating Validation Set for data in range [116776, 233550]
    -------- Start Training on 1574188362.94751
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 17.883373975753784
    ---- Creating Validation Set for data in range [233551, 350325]
    -------- Start Training on 1574188381.4168372
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 17.50961685180664
    ---- Creating Validation Set for data in range [350326, 467100]
    -------- Start Training on 1574188399.50402
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 17.69806694984436
    0
    	predicted
    		0.1315
    	avocado
    		0.1523
    	curr_impute
    		0.3132
    1
    	predicted
    		0.1315
    	avocado
    		0.1493
    	curr_impute
    		0.3041
    2
    	predicted
    		0.1337
    	avocado
    		0.1541
    	curr_impute
    		0.3129
    3
    	predicted
    		0.1332
    	avocado
    		0.1519
    	curr_impute
    		0.3134



```python
m21_result_wt_ideas = hist_models("M21", ml_df_shuf, False)
```

    ---- Creating Validation Set for data in range [0, 116775]
    -------- Start Training on 1574188997.884465
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 16.571380853652954
    ---- Creating Validation Set for data in range [116776, 233550]
    -------- Start Training on 1574189015.0444632
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 16.519545793533325
    ---- Creating Validation Set for data in range [233551, 350325]
    -------- Start Training on 1574189032.130722
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 16.46050000190735
    ---- Creating Validation Set for data in range [350326, 467100]
    -------- Start Training on 1574189049.203674
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 17.15988779067993
    0
    	predicted
    		0.1319
    	avocado
    		0.1523
    	curr_impute
    		0.3132
    1
    	predicted
    		0.1318
    	avocado
    		0.1493
    	curr_impute
    		0.3041
    2
    	predicted
    		0.1339
    	avocado
    		0.1541
    	curr_impute
    		0.3129
    3
    	predicted
    		0.1332
    	avocado
    		0.1519
    	curr_impute
    		0.3134


### Model for M03 - H2AFZ


```python
m03_result = hist_models("M03", ml_df_shuf)
```

    ---- Creating Validation Set for data in range [0, 93420]
    -------- Start Training on 1574188493.600275
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 13.731431722640991
    ---- Creating Validation Set for data in range [93421, 186840]
    -------- Start Training on 1574188507.847035
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 12.977031230926514
    ---- Creating Validation Set for data in range [186841, 280260]
    -------- Start Training on 1574188521.262924
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 12.248181104660034
    ---- Creating Validation Set for data in range [280261, 373680]
    -------- Start Training on 1574188533.940921
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 12.40692687034607
    0
    	predicted
    		0.1064
    	avocado
    		0.1181
    	curr_impute
    		0.2546
    1
    	predicted
    		0.1058
    	avocado
    		0.1173
    	curr_impute
    		0.255
    2
    	predicted
    		0.1081
    	avocado
    		0.1195
    	curr_impute
    		0.2601
    3
    	predicted
    		0.1075
    	avocado
    		0.1199
    	curr_impute
    		0.2613



```python
m03_result_wt_ideas = hist_models("M03", ml_df_shuf, False)
```

    ---- Creating Validation Set for data in range [0, 93420]
    -------- Start Training on 1574188889.4063501
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 12.069170951843262
    ---- Creating Validation Set for data in range [93421, 186840]
    -------- Start Training on 1574188901.90952
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 11.77048110961914
    ---- Creating Validation Set for data in range [186841, 280260]
    -------- Start Training on 1574188914.099287
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 11.769668817520142
    ---- Creating Validation Set for data in range [280261, 373680]
    -------- Start Training on 1574188926.2890701
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 11.894800901412964
    0
    	predicted
    		0.1067
    	avocado
    		0.1181
    	curr_impute
    		0.2546
    1
    	predicted
    		0.1053
    	avocado
    		0.1173
    	curr_impute
    		0.255
    2
    	predicted
    		0.1079
    	avocado
    		0.1195
    	curr_impute
    		0.2601
    3
    	predicted
    		0.1077
    	avocado
    		0.1199
    	curr_impute
    		0.2613


### Model for M25 - H3K79me2


```python
m25_result = hist_models("M25", ml_df_shuf)
```

    ---- Creating Validation Set for data in range [0, 93420]
    -------- Start Training on 1574188633.599432
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 13.147719860076904
    ---- Creating Validation Set for data in range [93421, 186840]
    -------- Start Training on 1574188647.193089
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 12.824733972549438
    ---- Creating Validation Set for data in range [186841, 280260]
    -------- Start Training on 1574188660.440157
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 12.775397062301636
    ---- Creating Validation Set for data in range [280261, 373680]
    -------- Start Training on 1574188673.624012
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 12.997114896774292
    0
    	predicted
    		0.1729
    	avocado
    		0.1953
    	curr_impute
    		0.2664
    1
    	predicted
    		0.1769
    	avocado
    		0.2005
    	curr_impute
    		0.2736
    2
    	predicted
    		0.1698
    	avocado
    		0.1891
    	curr_impute
    		0.2577
    3
    	predicted
    		0.1713
    	avocado
    		0.192
    	curr_impute
    		0.2624



```python
m25_result_wt_ideas = hist_models("M25", ml_df_shuf, False)
```

    ---- Creating Validation Set for data in range [0, 93420]
    -------- Start Training on 1574189102.0591948
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 11.923866271972656
    ---- Creating Validation Set for data in range [93421, 186840]
    -------- Start Training on 1574189114.4053
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 11.734225988388062
    ---- Creating Validation Set for data in range [186841, 280260]
    -------- Start Training on 1574189126.537584
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 11.834797859191895
    ---- Creating Validation Set for data in range [280261, 373680]
    -------- Start Training on 1574189138.7640052
    /Users/Michavillson/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    -------- Finished Training, elapsed time: 13.448953866958618
    0
    	predicted
    		0.1733
    	avocado
    		0.1953
    	curr_impute
    		0.2664
    1
    	predicted
    		0.1786
    	avocado
    		0.2005
    	curr_impute
    		0.2736
    2
    	predicted
    		0.1724
    	avocado
    		0.1891
    	curr_impute
    		0.2577
    3
    	predicted
    		0.1714
    	avocado
    		0.192
    	curr_impute
    		0.2624


## Convert the Above Statistics into Tidy Data Format


```python
print(marks_list)
```

    ['M21' 'M03' 'M25']



```python
hist_list = [["Mark_id", "Iteration", "predicted", "avocado", "curr_impute", "ideas"]]
M25_list =  [["M25", i[0], i[1]["predicted"], i[1]["avocado"], i[1]["curr_impute"], False] for i in m25_result_wt_ideas["MSE"].items()]
M25_list += [["M25", i[0], i[1]["predicted"], i[1]["avocado"], i[1]["curr_impute"], True] for i in m25_result["MSE"].items()]
M03_list =  [["M03", i[0], i[1]["predicted"], i[1]["avocado"], i[1]["curr_impute"], False] for i in m03_result_wt_ideas["MSE"].items()]
M03_list += [["M03", i[0], i[1]["predicted"], i[1]["avocado"], i[1]["curr_impute"], True] for i in m03_result["MSE"].items()]
M21_list =  [["M21", i[0], i[1]["predicted"], i[1]["avocado"], i[1]["curr_impute"], False] for i in m21_result_wt_ideas["MSE"].items()]
M21_list += [["M21", i[0], i[1]["predicted"], i[1]["avocado"], i[1]["curr_impute"], True] for i in m21_result["MSE"].items()]
hist_df = pd.DataFrame(M25_list + M03_list + M21_list, columns = hist_list[0])
```


```python
hist_df.to_csv("sources/ML_model/output/mse_hist_marks_model.csv", index=False)
hist_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mark_id</th>
      <th>Iteration</th>
      <th>predicted</th>
      <th>avocado</th>
      <th>curr_impute</th>
      <th>ideas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M25</td>
      <td>0</td>
      <td>0.1733</td>
      <td>0.1953</td>
      <td>0.2664</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M25</td>
      <td>1</td>
      <td>0.1786</td>
      <td>0.2005</td>
      <td>0.2736</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M25</td>
      <td>2</td>
      <td>0.1724</td>
      <td>0.1891</td>
      <td>0.2577</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M25</td>
      <td>3</td>
      <td>0.1714</td>
      <td>0.1920</td>
      <td>0.2624</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M25</td>
      <td>0</td>
      <td>0.1729</td>
      <td>0.1953</td>
      <td>0.2664</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M25</td>
      <td>1</td>
      <td>0.1769</td>
      <td>0.2005</td>
      <td>0.2736</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M25</td>
      <td>2</td>
      <td>0.1698</td>
      <td>0.1891</td>
      <td>0.2577</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M25</td>
      <td>3</td>
      <td>0.1713</td>
      <td>0.1920</td>
      <td>0.2624</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M03</td>
      <td>0</td>
      <td>0.1067</td>
      <td>0.1181</td>
      <td>0.2546</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M03</td>
      <td>1</td>
      <td>0.1053</td>
      <td>0.1173</td>
      <td>0.2550</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M03</td>
      <td>2</td>
      <td>0.1079</td>
      <td>0.1195</td>
      <td>0.2601</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M03</td>
      <td>3</td>
      <td>0.1077</td>
      <td>0.1199</td>
      <td>0.2613</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M03</td>
      <td>0</td>
      <td>0.1064</td>
      <td>0.1181</td>
      <td>0.2546</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M03</td>
      <td>1</td>
      <td>0.1058</td>
      <td>0.1173</td>
      <td>0.2550</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M03</td>
      <td>2</td>
      <td>0.1081</td>
      <td>0.1195</td>
      <td>0.2601</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M03</td>
      <td>3</td>
      <td>0.1075</td>
      <td>0.1199</td>
      <td>0.2613</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M21</td>
      <td>0</td>
      <td>0.1319</td>
      <td>0.1523</td>
      <td>0.3132</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M21</td>
      <td>1</td>
      <td>0.1318</td>
      <td>0.1493</td>
      <td>0.3041</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M21</td>
      <td>2</td>
      <td>0.1339</td>
      <td>0.1541</td>
      <td>0.3129</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M21</td>
      <td>3</td>
      <td>0.1332</td>
      <td>0.1519</td>
      <td>0.3134</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>M21</td>
      <td>0</td>
      <td>0.1315</td>
      <td>0.1523</td>
      <td>0.3132</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>M21</td>
      <td>1</td>
      <td>0.1315</td>
      <td>0.1493</td>
      <td>0.3041</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>M21</td>
      <td>2</td>
      <td>0.1337</td>
      <td>0.1541</td>
      <td>0.3129</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>M21</td>
      <td>3</td>
      <td>0.1332</td>
      <td>0.1519</td>
      <td>0.3134</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
