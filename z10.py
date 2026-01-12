import pandas as pd

data = {
 'Height': [170, 165, 180, 175, 160, 172, 168, 177, 162, 158],
 'Weight': [65, 59, 75, 68, 55, 70, 62, 74, 58, 54],
 'Age': [30, 25, 35, 28, 22, 32, 27, 33, 24, 21],
 'Gender': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0] # 1 = Male, 0 = Female
}

df = pd.DataFrame(data)
print(df)
