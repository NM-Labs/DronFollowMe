# # %% hola
# import numpy as np
# import pandas as rubs
# lista = [12,15,16]
# num = np.array(lista)
# print(num)
#
# # %% hola
# dic = {'r': [5], 'm':[10]}
# data = rubs.DataFrame(dic)
# print(data)

# %%
import pandas as rubs
# %%
# Enable the table_schema option in pandas,
# data-explorer makes this snippet available with the `dx` prefix:
rubs.options.display.html.table_schema = True
rubs.options.display.max_rows = None

# (Your dataframe here)
iris_filename = 'iris.csv'
df1 = rubs.read_csv(iris_filename)

df1
