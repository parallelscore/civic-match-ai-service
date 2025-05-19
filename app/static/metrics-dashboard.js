// API endpoint for the metrics JSON data
const API_ENDPOINT = '/json';

// DOM Elements
const loadingContainer = document.getElementById('loadingContainer');
const errorContainer = document.getElementById('errorContainer');
const dashboardContent = document.getElementById('dashboardContent');
const refreshButton = document.getElementById('refreshButton');
const refreshIcon = document.getElementById('refreshIcon');
const tryAgainButton = document.getElementById('tryAgainButton');
const lastUpdatedElement = document.getElementById('lastUpdated');
const errorMessageElement = document.getElementById('errorMessage');

// Helper Functions
function getColorClasses(percent) {
    if (percent < 50) return {
        valueClass: 'value-green',
        barClass: 'progress-green'
    };
    if (percent < 75) return {
        valueClass: 'value-orange',
        barClass: 'progress-orange'
    };
    return {
        valueClass: 'value-red',
        barClass: 'progress-red'
    };
}

function updateProgressBar(percentElement, barElement, percentage) {
    const colors = getColorClasses(percentage);

    percentElement.className = colors.valueClass;
    percentElement.textContent = `${percentage.toFixed(2)}%`;

    barElement.className = `progress-bar ${colors.barClass}`;
    barElement.style.width = `${Math.min(percentage, 100)}%`;
}

function showLoading() {
    loadingContainer.style.display = 'flex';
    errorContainer.style.display = 'none';
    dashboardContent.style.display = 'none';
    refreshButton.disabled = true;
}

function showError(message) {
    loadingContainer.style.display = 'none';
    errorContainer.style.display = 'flex';
    dashboardContent.style.display = 'none';
    errorMessageElement.textContent = message;
    refreshButton.disabled = false;
}

function showDashboard() {
    loadingContainer.style.display = 'none';
    errorContainer.style.display = 'none';
    dashboardContent.style.display = 'block';
    refreshButton.disabled = false;
}

// Start Refresh Animation
function startRefreshAnimation() {
    refreshButton.disabled = true;
    let rotation = 0;
    const animationInterval = setInterval(() => {
        rotation += 30;
        refreshIcon.style.display = 'inline-block';
        refreshIcon.style.transform = `rotate(${rotation}deg)`;
    }, 100);

    return animationInterval;
}

// Stop Refresh Animation
function stopRefreshAnimation(intervalId) {
    clearInterval(intervalId);
    refreshIcon.style.transform = 'rotate(0deg)';
    refreshButton.disabled = false;
}

// Fetch Metrics Function
async function fetchMetrics() {
    const animationInterval = startRefreshAnimation();

    try {
        showLoading();

        const response = await fetch(API_ENDPOINT);

        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        const data = await response.json();
        updateDashboard(data);
        lastUpdatedElement.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
        showDashboard();

    } catch (error) {
        console.error('Error fetching metrics:', error);
        showError(`Failed to fetch server metrics: ${error.message}`);
    } finally {
        stopRefreshAnimation(animationInterval);
    }
}

// Update Dashboard Function
function updateDashboard(metrics) {
    // Top Metrics
    document.getElementById('serverTime').textContent = metrics.currentServerTime ? new Date(metrics.currentServerTime).toLocaleTimeString() : 'N/A';
    document.getElementById('serverUptime').textContent = metrics.serverUptime || 'N/A';
    document.getElementById('requestsProcessed').textContent = metrics.requestsProcessed ? metrics.requestsProcessed.toLocaleString() : 'N/A';
    document.getElementById('activeProcesses').textContent = metrics.activeProcesses || 'N/A';

    // CPU Usage
    const cpuPercent = metrics.cpuUsage || 0;
    updateProgressBar(
        document.getElementById('cpuUsagePercent'),
        document.getElementById('cpuUsageBar'),
        cpuPercent
    );

    // CPU Cores
    const cpuCoresContainer = document.getElementById('cpuCoresContainer');
    cpuCoresContainer.innerHTML = '';

    if (metrics.cpuUsagePerCore && metrics.cpuUsagePerCore.length > 0) {
        metrics.cpuUsagePerCore.forEach((core) => {
            const coreElement = document.createElement('div');
            coreElement.className = 'inner-card';

            const colors = getColorClasses(core.usage || 0);

            coreElement.innerHTML = `
                <h4 class="inner-card-title">Core ${core.core}</h4>
                <p class="inner-card-value">${core.usage.toFixed(2)}%</p>
                <div class="progress-container">
                    <div class="progress-bar ${colors.barClass}" style="width: ${Math.min(core.usage, 100)}%"></div>
                </div>
            `;

            cpuCoresContainer.appendChild(coreElement);
        });
    }

    // Load Average
    document.getElementById('loadAvg1Min').textContent = metrics.loadAverage ? metrics.loadAverage.oneMin.toFixed(2) : 'N/A';
    document.getElementById('loadAvg5Min').textContent = metrics.loadAverage ? metrics.loadAverage.fiveMin.toFixed(2) : 'N/A';
    document.getElementById('loadAvg15Min').textContent = metrics.loadAverage ? metrics.loadAverage.fifteenMin.toFixed(2) : 'N/A';

    // Memory Usage
    const memoryPercent = metrics.memoryInfo ? metrics.memoryInfo.percentUsed : 0;
    updateProgressBar(
        document.getElementById('memoryPercent'),
        document.getElementById('memoryBar'),
        memoryPercent
    );

    document.getElementById('memoryTotal').textContent = metrics.memoryInfo ? `${metrics.memoryInfo.total.toFixed(2)} MB` : 'N/A';
    document.getElementById('memoryUsed').textContent = metrics.memoryInfo ? `${metrics.memoryInfo.used.toFixed(2)} MB` : 'N/A';
    document.getElementById('memoryAvailable').textContent = metrics.memoryInfo ? `${metrics.memoryInfo.available.toFixed(2)} MB` : 'N/A';
    document.getElementById('memoryFree').textContent = metrics.memoryInfo ? `${metrics.memoryInfo.free.toFixed(2)} MB` : 'N/A';

    // Swap Memory
    const swapPercent = metrics.swapMemoryInfo ? metrics.swapMemoryInfo.percentUsed : 0;
    updateProgressBar(
        document.getElementById('swapPercent'),
        document.getElementById('swapBar'),
        swapPercent
    );

    document.getElementById('swapTotal').textContent = metrics.swapMemoryInfo ? `${metrics.swapMemoryInfo.total.toFixed(2)} MB` : 'N/A';
    document.getElementById('swapUsed').textContent = metrics.swapMemoryInfo ? `${metrics.swapMemoryInfo.used.toFixed(2)} MB` : 'N/A';
    document.getElementById('swapFree').textContent = metrics.swapMemoryInfo ? `${metrics.swapMemoryInfo.free.toFixed(2)} MB` : 'N/A';

    // Disk Usage
    const diskPercent = metrics.diskUsage ? metrics.diskUsage.percentUsed : 0;
    updateProgressBar(
        document.getElementById('diskPercent'),
        document.getElementById('diskBar'),
        diskPercent
    );

    document.getElementById('diskTotal').textContent = metrics.diskUsage ? `${metrics.diskUsage.total.toFixed(2)} GB` : 'N/A';
    document.getElementById('diskUsed').textContent = metrics.diskUsage ? `${metrics.diskUsage.used.toFixed(2)} GB` : 'N/A';
    document.getElementById('diskFree').textContent = metrics.diskUsage ? `${metrics.diskUsage.free.toFixed(2)} GB` : 'N/A';

    // Disk I/O
    document.getElementById('diskRead').textContent = metrics.diskIO ? `${metrics.diskIO.read.toFixed(2)} MB` : 'N/A';
    document.getElementById('diskWrite').textContent = metrics.diskIO ? `${metrics.diskIO.write.toFixed(2)} MB` : 'N/A';

    // Network I/O
    document.getElementById('networkSent').textContent = metrics.networkIO ? `${metrics.networkIO.bytesSent.toFixed(2)} MB` : 'N/A';
    document.getElementById('networkReceived').textContent = metrics.networkIO ? `${metrics.networkIO.bytesReceived.toFixed(2)} MB` : 'N/A';

}

// Event Listeners
refreshButton.addEventListener('click', fetchMetrics);
tryAgainButton.addEventListener('click', fetchMetrics);

// Initialize the dashboard on a page load
document.addEventListener('DOMContentLoaded', fetchMetrics);
