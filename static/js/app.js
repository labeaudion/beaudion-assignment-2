const svg = d3.select("#plot");
let points = []; // Array to store generated data points
let selectedCentroids = []; // Array to store manual centroids
let currentLabels = []; // To store the labels from KMeans
let centroids = []; // To store the centroids from KMeans
let steps = []; // Array to store steps for step-through
let currentStep = 0; // To keep track of the current step
let isKmeansStarted = false; // Track if KMeans has been started


// Function to generate random data points
function generateData() {
    points = [];
    for (let i = 0; i < 100; i++) {
        points.push([Math.random() * 600, Math.random() * 400]);
    }
    drawPoints();
}

// Function to draw points on the SVG
function drawPoints() {
    svg.selectAll("circle.data-point").remove(); // Clear previous points
    svg.selectAll("circle.data-point")
        .data(points)
        .enter()
        .append("circle")
        .attr("class", "data-point")
        .attr("cx", d => d[0])
        .attr("cy", d => d[1])
        .attr("r", 5)
        .attr("fill", "blue");
}

// Capture clicks for manual centroid selection
function enableManualSelection() {
    svg.on("click", function(event) {
        const coords = d3.pointer(event);
        const maxCentroids = parseInt(document.getElementById('numCentroids').value, 10);
        if (selectedCentroids.length < maxCentroids) { // Limit to the user-defined number of clusters
            selectedCentroids.push(coords); // Store clicked coordinates
            drawCentroid(coords); // Visualize the selected centroid
        } else {
            alert(`You can only select up to ${maxCentroids} centroids.`);
        }
    });
}

function disableManualSelection() {
    svg.on("click", null); // Remove click event listener
}

// Function to draw the selected centroid
function drawCentroid(coords) {
    svg.append("circle")
        .attr("cx", coords[0])
        .attr("cy", coords[1])
        .attr("r", 12)
        .attr("fill", "red")
        .attr("class", "centroid")
        .attr("stroke", "black") // Add a black stroke for better visibility
        .attr("stroke-width", 2); // Set stroke width
}

// Event listener for convergence button
document.getElementById('converge-kmeans').addEventListener('click', () => {
    if (!isKMeansStarted) {
        alert("Please start KMeans first.");
        return;
    }

    // Visualize the final state after convergence
    drawStep(steps[steps.length - 1]); // Draw the last step (final result)
});

// Function to visualize the initial centroids
function visualizeInitialCentroids() {
    svg.selectAll("circle.centroid").remove(); // Clear previous centroids

    // Draw the resulting centroids
    centroids.forEach((centroid, index) => {
        svg.append("circle")
            .attr("cx", centroid[0])
            .attr("cy", centroid[1])
            .attr("r", 8)
            .attr("fill", d3.schemeCategory10[index % 10]) // Use color scheme
            .attr("class", "centroid");
    });

    // Optionally draw initial data points with their initial labels
    points.forEach((point, index) => {
        svg.append("circle")
            .attr("cx", point[0])
            .attr("cy", point[1])
            .attr("r", 5)
            .attr("fill", d3.schemeCategory10[currentLabels[index] % 10]) // Color by initial cluster
            .attr("class", "result-point");
    });
}

// Run KMeans and show the first step
document.getElementById('start-kmeans').addEventListener('click', async () => {
    const method = document.getElementById('init-method').value;

    // Check for manual method and selected centroids
    if (method === 'manual' && selectedCentroids.length === 0) {
        alert("Please select centroids by clicking on the graph.");
        return;
    }

    // Prepare data to send to the server
    const requestData = {
        data: points,
        init_method: method,
    };

    // Include selected centroids only for manual method
    if (method === 'manual') {
        requestData.initial_centroids = selectedCentroids;
    }

    const response = await fetch('/kmeans', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    });

    const result = await response.json();
    centroids = result.centroids; // Store the new centroids
    currentLabels = result.labels; // Store the labels
    steps = result.steps; // Store the steps for step-through

    // Visualize the first step
    drawStep(steps[0]); // Draw the first step
    isKMeansStarted = true; // Set KMeans started flag
    alert("KMeans has started. You can now step through or converge to the final result.");
});

// Function to visualize the KMeans results
function visualizeResults() {
    svg.selectAll("circle.result-point").remove(); // Clear previous results
    svg.selectAll("circle.centroid").remove(); // Clear previous centroids

    // Draw the resulting centroids
    centroids.forEach((centroid, index) => {
        svg.append("circle")
            .attr("cx", centroid[0])
            .attr("cy", centroid[1])
            .attr("r", 8)
            .attr("fill", d3.schemeCategory10[index % 10]) // Use color scheme
            .attr("class", "centroid");
    });

    // Draw the resulting data points based on their labels
    points.forEach((point, index) => {
        svg.append("circle")
            .attr("cx", point[0])
            .attr("cy", point[1])
            .attr("r", 5)
            .attr("fill", d3.schemeCategory10[currentLabels[index] % 10]) // Color by cluster
            .attr("class", "result-point");
    });
}

// Step through the KMeans steps
document.getElementById('step-kmeans').addEventListener('click', () => {
    if (currentStep < steps.length) {
        const step = steps[currentStep];
        drawStep(step);
        currentStep++;
    } else {
        alert("No more steps to display.");
    }
});

// Function to draw a specific step
function drawStep(step) {
    svg.selectAll("circle.result-point").remove(); // Clear previous results
    svg.selectAll("circle.centroid").remove(); // Clear previous centroids

    // Draw centroids for this step
    step.centroids.forEach((centroid, index) => {
        svg.append("circle")
            .attr("cx", centroid[0])
            .attr("cy", centroid[1])
            .attr("r", 12)
            .attr("fill", d3.schemeCategory10[index % 10]) // Use color scheme
            .attr("class", "centroid")
            .attr("stroke", "black") // Add a black stroke for better visibility
            .attr("stroke-width", 2); // Set stroke width
    });

    // Draw data points for this step
    points.forEach((point, index) => {
        svg.append("circle")
            .attr("cx", point[0])
            .attr("cy", point[1])
            .attr("r", 5)
            .attr("fill", d3.schemeCategory10[step.labels[index] % 10]) // Color by cluster
            .attr("class", "result-point");
    });
}

// Reset the algorithm and clear selected centroids
document.getElementById('reset').addEventListener('click', () => {
    selectedCentroids = []; // Clear selected centroids
    svg.selectAll("circle.centroid").remove(); // Remove centroids from visualization
    svg.selectAll("circle.result-point").remove(); // Clear result points
    currentLabels = []; // Clear current labels
    centroids = []; // Clear centroids
    steps = []; // Clear steps for step-through
    currentStep = 0; // Reset the step counter
    alert("Algorithm reset. You can choose new centroids if manual mode is selected.");
});

// Initialize the app
document.getElementById('generate').addEventListener('click', generateData);

// Event listener for initialization method selection
document.getElementById('init-method').addEventListener('change', (event) => {
    if (event.target.value === 'manual') {
        enableManualSelection(); // Enable centroid selection
    } else {
        disableManualSelection(); // Disable centroid selection
        selectedCentroids = []; // Clear previously selected centroids
        svg.selectAll("circle.centroid").remove(); // Clear centroids from visualization
    }
});
