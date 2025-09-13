% Search and Rescue Simulation - Initial Setup

% Clear workspace and command window
clear;
clc;

% Define grid size
gridSize = 50;

% Create the environment (1 = free space, 0 = obstacle)
environment = ones(gridSize);

% Add some random obstacles
numObstacles = 50;
obstacleIndices = randi([1, gridSize^2], 1, numObstacles);
environment(obstacleIndices) = 0;

% Initialize robot positions (example with 3 robots)
numRobots = 3;
robotPositions = randi([1, gridSize], numRobots, 2);

% Initialize victim positions (example with 5 victims)
numVictims = 5;
victimPositions = randi([1, gridSize], numVictims, 2);

% Visualization
figure;
hold on;

% Plot environment
imagesc(environment);
colormap([1 0 0; 1 1 1]); % Red for obstacles, white for free space

% Plot robots
robotColors = {'b', 'g', 'y'}; % Blue, green, yellow for robots
for i = 1:numRobots
    plot(robotPositions(i,2), robotPositions(i,1), 'o', 'MarkerSize', 10, 'MarkerFaceColor', robotColors{i}, 'MarkerEdgeColor', 'k');
end

% Plot victims
plot(victimPositions(:,2), victimPositions(:,1), 'p', 'MarkerSize', 10, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k');

% Set up the plot
axis equal;
axis([0.5 gridSize+0.5 0.5 gridSize+0.5]);
title('Search and Rescue Simulation');
xlabel('X');
ylabel('Y');

legend('Obstacle', 'Free Space', 'Robot 1', 'Robot 2', 'Robot 3', 'Victim', 'Location', 'eastoutside');

hold off;