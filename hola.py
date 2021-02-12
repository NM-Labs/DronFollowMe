# # %% hola
# import numpy as np
# import pandas as pd
# lista = [12,15,16]
# num = np.array(lista)
# print(num)
#
# # %% hola
# dic = {'r': [5], 'm':[10]}
# data = pd.DataFrame(dic)
# print(data)

# %%
import pandas as pd
# %%
# Enable the table_schema option in pandas,
# data-explorer makes this snippet available with the `dx` prefix:
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

# (Your dataframe here)
iris_filename = 'iris.csv'
df1 = pd.read_csv(iris_filename)

df1
