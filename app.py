from flask import Flask, render_template, request, jsonify
import numpy as np
from kmeans import KMeans

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/kmeans', methods=['POST'])
def kmeans():
    data = request.json['data']
    n_clusters = request.json['n_clusters']
    init_method = request.json['init_method']

    kmeans = KMeans(n_clusters=n_clusters, init_method=init_method)
    kmeans.fit(np.array(data))

    response = {
        'labels': kmeans.labels.tolist(),
        'centroids': kmeans.centroids.tolist(),
        'history': [(labels.tolist(), centroids.tolist()) for labels, centroids in kmeans.history]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
