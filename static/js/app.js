const svg = d3.select("#plot")
    .attr("width", 700)
    .attr("height", 500);
let points = []; // Array to store generated data points
let selectedCentroids = []; // Array to store manual centroids
let currentLabels = []; // To store the labels from KMeans
let centroids = []; // To store the centroids from KMeans
let steps = []; // Array to store steps for step-through
let currentStep = 0; // To keep track of the current step
let isKmeansStarted = false; // Track if KMeans has been started


const xScale = d3.scaleLinear()
    .domain([0, 600])
    .range([50, 650]);

const yScale = d3.scaleLinear()
    .domain([0, 400])
    .range([450, 50]);

const xAxis = d3.axisBottom(xScale);
svg.append("g")
    .attr("transform", "translate(0, 450)") // Position the x-axis
    .call(xAxis);

const yAxis = d3.axisLeft(yScale);
svg.append("g")
    .attr("transform", "translate(50, 0)") // Position the y-axis
    .call(yAxis);


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
        .attr("cx", d => xScale(d[0]))
        .attr("cy", d => yScale(d[1]))
        .attr("r", 5)
        .attr("fill", "blue");
}



// Capture clicks for manual centroid selection
function enableManualSelection() {
    svg.on("click", function(event) {
        const coords = d3.pointer(event);
        const clampedX = Math.max(50, Math.min(650, coords[0]));
        const clampedY = Math.max(50, Math.min(450, coords[1]));
        const maxCentroids = parseInt(document.getElementById('numCentroids').value, 10);
        if (selectedCentroids.length < maxCentroids) { // Limit to the user-defined number of clusters
            selectedCentroids.push([clampedX, clampedY]); // Store clicked coordinates
            drawCentroid([clampedX, clampedY]); // Visualize the selected centroid
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

    // Alert the user that the algorithm has converged
    alert("Kmeans has converged!")
});

// Function to visualize the initial centroids
function visualizeInitialCentroids() {
    svg.selectAll("circle.centroid").remove(); // Clear previous centroids

    // Draw the resulting centroids
    centroids.forEach((centroid, index) => {
        svg.append("circle")
            .attr("cx", xScale(centroid[0]))
            .attr("cy", yScale(centroid[1]))
            .attr("r", 8)
            .attr("fill", d3.schemeCategory10[index % 10]) // Use color scheme
            .attr("class", "centroid");
    });

    // Optionally draw initial data points with their initial labels
    points.forEach((point, index) => {
        svg.append("circle")
            .attr("cx", xScale(point[0]))
            .attr("cy", yScale(point[1]))
            .attr("r", 5)
            .attr("fill", d3.schemeCategory10[currentLabels[index] % 10]) // Color by initial cluster
            .attr("class", "result-point");
    });
}

// Run KMeans and show the first step
document.getElementById('start-kmeans').addEventListener('click', async () => {
    const method = document.getElementById('init-method').value;
    const nClusters = parseInt(document.getElementById('numCentroids').value, 10); // Fetch the number again

    // Check for manual method and selected centroids
    if (method === 'manual' && selectedCentroids.length === 0) {
        alert("Please select centroids by clicking on the graph.");
        return;
    }

    // Prepare data to send to the server
    const requestData = {
        data: points,
        init_method: method,
        n_clusters: nClusters
    };

    // Include selected centroids only for manual method
    if (method === 'manual') {
        // Send original coordinates, converting back to original range
        requestData.initial_centroids = selectedCentroids.map(coord => [
            (coord[0] - 50),  // Reverse scaling by subtracting the offset
            (450 - coord[1])  // Reverse scaling for Y-axis
        ]);
    
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
            .attr("cx", xScale(centroid[0]))
            .attr("cy", yScale(centroid[1]))
            .attr("r", 8)
            .attr("fill", d3.schemeCategory10[index % 10]) // Use color scheme
            .attr("class", "centroid");
    });

    // Draw the resulting data points based on their labels
    points.forEach((point, index) => {
        svg.append("circle")
            .attr("cx", xScale(point[0]))
            .attr("cy", yScale(point[1]))
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
        alert("No more steps to display, Kmeans has converged.");
    }
});

// Function to draw a specific step
function drawStep(step) {
    svg.selectAll("circle.result-point").remove(); // Clear previous results
    svg.selectAll("circle.centroid").remove(); // Clear previous centroids

    // Draw centroids for this step
    step.centroids.forEach((centroid, index) => {
        svg.append("circle")
            .attr("cx", xScale(centroid[0]))
            .attr("cy", yScale(centroid[1]))
            .attr("r", 12)
            .attr("fill", d3.schemeCategory10[index % 10]) // Use color scheme
            .attr("class", "centroid")
            .attr("stroke", "black") // Add a black stroke for better visibility
            .attr("stroke-width", 2); // Set stroke width
    });

    // Draw data points for this step
    points.forEach((point, index) => {
        svg.append("circle")
            .attr("cx", xScale(point[0]))
            .attr("cy", yScale(point[1]))
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
