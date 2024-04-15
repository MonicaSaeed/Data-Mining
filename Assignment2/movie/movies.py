import pandas as pd

def read_data(file_path, percentage):
    num_rows = sum(1 for line in open(file_path)) 
    num_read_rows = int(num_rows * percentage / 100)
    data = pd.read_csv(file_path, usecols=['Movie Name', 'IMDB Rating'], nrows=num_read_rows)
    data = data.sort_values(by='IMDB Rating')
    return data

def get_outlier(data):
    # set first 10% of the data and last 10% as outliers
    outliers = pd.concat([data.head(int(len(data) * 0.1)), data.tail(int(len(data) * 0.1))])
    return outliers

def remove_outlier(data):
    # remove the first 10% of the data
    data = data.iloc[int(len(data) * 0.1):]
    # remove the last 10% of the data
    data = data.iloc[:int(len(data) * 0.9)]
    return data

def distance(x, y):
    return abs(x - y)

def set_centroids(row, centroids):
    distances = [distance(row, centroid) for centroid in centroids]
    return distances.index(min(distances))

def k_means(data, centroids):
    clusters = {}
    check = True
    while check:
        for i in range(len(centroids)):
            clusters[i] = {'Movie Name': [], 'IMDB Rating': []}  # Initialize each cluster with lists for movie names and IMDB ratings
        for index, row in data.iterrows():
            cluster_index = set_centroids(row['IMDB Rating'], centroids)
            clusters[cluster_index]['Movie Name'].append(row['Movie Name'])  # Append movie name to the corresponding cluster
            clusters[cluster_index]['IMDB Rating'].append(row['IMDB Rating'])  # Append IMDB rating to the corresponding cluster
        new_centroids = [sum(clusters[i]['IMDB Rating']) / len(clusters[i]['IMDB Rating']) for i in clusters]
        if new_centroids == centroids:
            check = False
        else:
            centroids = new_centroids
    return clusters , centroids

file_path = r'D:\fcai\fourth level #2\DataMining\assignment\Data-Mining\Assignment2\movie\imdb_top_2000_movies.csv'
data = read_data(file_path, 1)
print(data)

outliers = get_outlier(data)
print(outliers)

data = remove_outlier(data)
print(data)

k = int(input('Enter the number of clusters: '))
# random first k centroids
centroids = [data.sample()['IMDB Rating'].iloc[0] for _ in range(k)]  
print(centroids)

clusters, centroids = k_means(data, centroids)

# loop for k and print centroids and thin the movies in each cluster
for i in range(k):
    print(f'Cluster {i + 1} Centroid: {centroids[i]}')
    print(clusters[i])
    