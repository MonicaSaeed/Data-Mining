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
            cluster_frame = tk.Frame(frame)
            cluster_frame.grid(row= 2 + (i // 2), column=i % 2, padx=10, pady=10)
            
            # Add a label for cluster centroid
            cluster_label = tk.Label(cluster_frame, text=f'Cluster {i + 1} Centroid: {centroids[i]}')
            cluster_label.pack()
            
            # Create a canvas for each cluster frame
            inner_canvas = tk.Canvas(cluster_frame, width=400, height=200)
            inner_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Add a scrollbar for each cluster canvas
            scrollbar = ttk.Scrollbar(cluster_frame, orient=tk.VERTICAL, command=inner_canvas.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            inner_canvas.configure(yscrollcommand=scrollbar.set)
            
            # Create a frame inside each cluster canvas to hold movie labels
            frame_in_canvas = tk.Frame(inner_canvas)
            inner_canvas.create_window((0,0), window=frame_in_canvas, anchor='nw')
            
            # Populate each frame with movie labels
            for j, movie in enumerate(cluster):
                movie_label = tk.Label(frame_in_canvas, text=str(movie))
                movie_label.pack()
        
        # add scrollbar to the canvas
        canvas = tk.Canvas(frame, width=800, height=200)
        canvas.grid(row=2 + (k // 2), column=0, columnspan=2)
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_outliers():
    try:
        file_path = entry_path.get()
        percentage = int(entry_percentage.get())
        
        data = read_data(file_path, percentage)
        outliers = get_outlier(data)
        
        # Create a new Toplevel window for displaying outliers
        outliers_window = tk.Toplevel()
        outliers_window.title("Outliers")
        outliers_window.geometry("700x700")  # Set the size of the window
        
        # Create a text widget for displaying outliers
        text_widget = tk.Text(outliers_window, wrap="word", width=80, height=20)
        text_widget.pack(fill="both", expand=True)
        
        # Add outliers data to the text widget
        text_widget.insert("1.0", outliers.to_string(index=False))
        
        # Add scrollbar to the text widget
        scrollbar = ttk.Scrollbar(text_widget, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        text_widget.config(yscrollcommand=scrollbar.set)
        
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Create GUI
root = tk.Tk()
root.title("K-means Clustering")
root.geometry("800x600")  # Adjust the initial size of the window

# Create a canvas to hold all content
canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a scrollbar to the canvas
scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame inside the canvas to hold all clusters and labels
frame = tk.Frame(canvas)
canvas.create_window((0,0), window=frame, anchor='nw')

# Function to update the canvas scrolling region
def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", on_configure)

# Create a label frame to group path, percentage, and clusters labels
label_frame = tk.LabelFrame(frame, text="Input Data")
label_frame.grid(row=0, column=0, padx=10, pady=10, columnspan=2, sticky='nsew')

label_path = tk.Label(label_frame, text="CSV File Path:")
label_path.grid(row=0, column=0, padx=5, pady=5)
entry_path = tk.Entry(label_frame, width=50)
entry_path.grid(row=0, column=1, padx=5, pady=5)
btn_browse = tk.Button(label_frame, text="Browse", command=open_file_dialog)
btn_browse.grid(row=0, column=2, padx=5, pady=5)

label_percentage = tk.Label(label_frame, text="Percentage of Data to Use:")
label_percentage.grid(row=1, column=0, padx=5, pady=5)
entry_percentage = tk.Entry(label_frame)
entry_percentage.grid(row=1, column=1, padx=5, pady=5)

label_clusters = tk.Label(label_frame, text="Number of Clusters:")
label_clusters.grid(row=2, column=0, padx=5, pady=5)
entry_clusters = tk.Entry(label_frame)
entry_clusters.grid(row=2, column=1, padx=5, pady=5)

btn_start = tk.Button(frame, text="Start Clustering", command=start_clustering)
btn_start.grid(row=1, column=0, padx=10, pady=10)

# Add the "Show Outliers" button
btn_outliers = tk.Button(frame, text="Show Outliers", command=show_outliers)
btn_outliers.grid(row=1, column=1, padx=10, pady=10)

# Update the scroll region of the canvas
frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

root.mainloop()
