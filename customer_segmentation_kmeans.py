#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read the dataset
df = pd.read_csv("C:/Users/Rahim Mahagonde/Downloads/Customer+Segmentation+Code+and+Files/Code and Files/train.csv")

# Create a Tkinter window
root = tk.Tk()
root.title("Customer Segmentation Analysis")
root.geometry("800x600")  # Set initial window size

# Create a style for tabs
style = ttk.Style()
style.theme_create("custom", parent="alt", settings={
    "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0]}},
    "TNotebook.Tab": {
        "configure": {"padding": [20, 5], "background": "#f0f0f0"},
        "map": {"background": [("selected", "#0078D4")]}
    }
})
style.theme_use("custom")

# Create a notebook (tabbed interface)
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# Create tabs
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
notebook.add(tab1, text="Data Analysis")
notebook.add(tab2, text="Clustering Analysis")

# Tab 1 - Data Analysis
label1 = tk.Label(tab1, text="Information about all data types:", font=("Helvetica", 14, "bold"))
label1.pack(pady=(10, 0))

# Display DataFrame in a Text widget
text_widget = tk.Text(tab1, height=10, width=80, font=("Helvetica", 12))
text_widget.insert("1.0", str(df.info()))
text_widget.pack()

# Display histograms using Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

for i, x in enumerate(['Age', 'Work_Experience', 'Family_Size']):
    axes[i].hist(df[x], bins=20)
    axes[i].set_title(x)

canvas = FigureCanvasTkAgg(fig, master=tab1)
canvas.get_tk_widget().pack()

# Tab 2 - Clustering Analysis
label2 = tk.Label(tab2, text="K-Means Clustering Analysis:", font=("Helvetica", 14, "bold"))
label2.pack(pady=(10, 0))

# Perform k-means clustering
X1 = df[['Age', 'Family_Size']].dropna().values
algorithm = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan')
algorithm.fit(X1)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

# Display k-means clustering results using Matplotlib
fig2 = plt.figure(figsize=(10, 6))
plt.scatter(X1[:, 0], X1[:, 1], c=labels2, s=50, cmap='viridis')
plt.scatter(centroids2[:, 0], centroids2[:, 1], marker='X', s=200, color='red')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Family Size', fontsize=12)
plt.title('K-Means Clustering', fontsize=14)
canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
canvas2.get_tk_widget().pack()

# Run the Tkinter event loop
root.mainloop()


# In[ ]:




