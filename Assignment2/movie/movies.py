import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd

class Movie:
    def __init__(self, name, rating):
        self.name = name
        self.rating = rating
    def __eq__(self, other):
        return self.rating == other.rating
    def __str__(self):
        return f'{self.name} {self.rating}'
    def __repr__(self):
        return self.__str__()

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

def assign_to_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for movie in data:
        centroid_index = set_centroids(movie.rating, centroids)
        clusters[centroid_index].append(movie)
    return clusters

def update_centroids(clusters):
    centroids = []
    for cluster in clusters:
        ratings = [movie.rating for movie in cluster]
        if ratings:
            centroids.append(sum(ratings) / len(ratings))
        else:
            centroids.append(0)  # handle empty clusters
    return centroids

def k_means(data, centroids):
    while True:
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:  # convergence check
            break
        else:
            centroids = new_centroids
    return clusters, centroids


# file_path = r'D:\fcai\fourth level #2\DataMining\assignment\Data-Mining\Assignment2\movie\imdb_top_2000_movies.csv'
# data = read_data(file_path,10)
# print(data)

# outliers = get_outlier(data)
# print(outliers)

# # data = remove_outlier(data)
# # print(data)

# k = int(input('Enter the number of clusters: '))
# # random first k centroids
# centroids = [data.sample()['IMDB Rating'].iloc[0] for _ in range(k)]  

# data = [Movie(row['Movie Name'], row['IMDB Rating']) for _, row in data.iterrows()]

# clusters, centroids = k_means(data, centroids)
# for i, cluster in enumerate(clusters):
#     print(f'Cluster {i + 1} Centroid: {centroids[i]}')
#     for movie in cluster:
#         print(movie)
#     print("-----------------------------")





# The rest of your code...
def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select CSV file")
    if file_path:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, file_path)

def start_clustering():
    try:
        file_path = entry_path.get()
        percentage = int(entry_percentage.get())
        k = int(entry_clusters.get())
        
        data = read_data(file_path, percentage)
        centroids = [data.sample()['IMDB Rating'].iloc[0] for _ in range(k)]
        data = [Movie(row['Movie Name'], row['IMDB Rating']) for _, row in data.iterrows()]
        
        clusters, centroids = k_means(data, centroids)
        
        for i, cluster in enumerate(clusters):
            # Create a separate frame for each cluster
            cluster_frame = tk.Frame(canvas)
            canvas.create_window((0, i * 200), window=cluster_frame, anchor='nw')
            
            # Add a label for cluster centroid
            cluster_label = tk.Label(cluster_frame, text=f'Cluster {i + 1} Centroid: {centroids[i]}')
            cluster_label.pack()
            
            # Create a canvas for each cluster frame
            inner_canvas = tk.Canvas(cluster_frame)
            inner_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Add a scrollbar for each cluster canvas
            scrollbar = ttk.Scrollbar(cluster_frame, orient=tk.VERTICAL, command=inner_canvas.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            inner_canvas.configure(yscrollcommand=scrollbar.set)
            inner_canvas.bind('<Configure>', lambda e, canvas=inner_canvas: canvas.configure(scrollregion=canvas.bbox("all")))
            
            # Create a frame inside each cluster canvas to hold movie labels
            frame = tk.Frame(inner_canvas)
            inner_canvas.create_window((0,0), window=frame, anchor='nw')
            
            # Populate each frame with movie labels
            for j, movie in enumerate(cluster):
                movie_label = tk.Label(frame, text=str(movie))
                movie_label.pack()
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create GUI
root = tk.Tk()
root.title("K-means Clustering")
root.geometry("800x750")

# Create a canvas with scrollbar
canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create a frame inside the canvas to hold all clusters
frame = tk.Frame(canvas)
canvas.create_window((0,0), window=frame, anchor='nw')

label_path = tk.Label(frame, text="CSV File Path:")
label_path.pack()
entry_path = tk.Entry(frame, width=50)
entry_path.pack()
btn_browse = tk.Button(frame, text="Browse", command=open_file_dialog)
btn_browse.pack()

label_percentage = tk.Label(frame, text="Percentage of Data to Use:")
label_percentage.pack()
entry_percentage = tk.Entry(frame)
entry_percentage.pack()

label_clusters = tk.Label(frame, text="Number of Clusters:")
label_clusters.pack()
entry_clusters = tk.Entry(frame)
entry_clusters.pack()

btn_start = tk.Button(frame, text="Start Clustering", command=start_clustering)
btn_start.pack()

root.mainloop()