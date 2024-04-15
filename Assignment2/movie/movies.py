import pandas as pd

def read_data(percentage):
    num_rows = 0
    with open('imdb_top_2000_movies.csv') as f:
        num_rows = sum(1 for line in f)
    num_read_rows = int(num_rows * percentage / 100)
    data = pd.read_csv('imdb_top_2000_movies.csv', usecols=['Movie Name', 'IMDB Rating'], nrows=num_read_rows)
    return data


data = read_data(2)
print(data)
