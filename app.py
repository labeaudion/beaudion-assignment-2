from flask import Flask, request, jsonify, render_template
import numpy as np
from kmeans import KMeans  # Importing the KMeans class from kmeans.py

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render HTML from templates directory

@app.route('/kmeans', methods=['POST'])
def run_kmeans():
    data = request.json['data']
    init_method = request.json['init_method']
    initial_centroids = request.json.get('initial_centroids')
    n_clusters = request.json.get('n_clusters', 3)  # Default to 3 if not provided

    # Convert input data to numpy array
    data = np.array(data)

    # Initialize KMeans and fit the model
    kmeans = KMeans(n_clusters=int(n_clusters))  # Convert to int
    kmeans.fit(data, init_method=init_method, initial_centroids=initial_centroids)

    return jsonify({
        'centroids': kmeans.centroids.tolist(),
        'labels': kmeans.labels.tolist(),
        'steps': kmeans.get_steps()
    })

if __name__ == '__main__':
    app.run(debug=True, port=3000)
