# %%
import pickle

with open("./data/tsp/tsp_5_train/tsp_5_reduced_train.pickle", 'rb') as file:
    data = pickle.load(file)

len(data['x_train'])
# %%
