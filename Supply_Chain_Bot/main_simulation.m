% SupplyChainBot Simulation with Performance Metrics

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

% Initialize robot positions and assign algorithms
numRobots = 4;
robotPositions = randi([1, gridSize], numRobots, 2);
robotAlgorithms = {'A*', 'Dijkstra', 'BFS', 'Genetic Hybrid'};
robotColors = {'b', 'g', 'y', 'm'};

% Define dynamic obstacle parameters
numDynamicObstacles = 3;
dynamicObstaclePositions = randi([1, gridSize], numDynamicObstacles, 2);
dynamicObstacleVelocities = (rand(numDynamicObstacles, 2) - 0.5) * 2; % Random velocities between -1 and 1

% Initialize supply points and delivery destinations
numSupplyPoints = 3;
numDeliveryPoints = 5;
supplyPointPositions = randi([1, gridSize], numSupplyPoints, 2);
deliveryPointPositions = randi([1, gridSize], numDeliveryPoints, 2);

% Create main figure
mainFigure = figure('Position', [100, 100, 800, 600]);

% Plot environment
imagesc(environment);
colormap([1 0 0; 1 1 1]); % Red for obstacles, white for free space
hold on;

% Plot robots
robotHandles = gobjects(1, numRobots);
for i = 1:numRobots
    robotHandles(i) = plot(robotPositions(i,2), robotPositions(i,1), 'o', 'MarkerSize', 10, 'MarkerFaceColor', robotColors{i}, 'MarkerEdgeColor', 'k');
end

% Plot supply points and delivery destinations
supplyPointHandle = plot(supplyPointPositions(:,2), supplyPointPositions(:,1), 's', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k');
deliveryPointHandle = plot(deliveryPointPositions(:,2), deliveryPointPositions(:,1), 'p', 'MarkerSize', 10, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k');

% Create dynamic obstacle handles
dynamicObstacleHandles = gobjects(1, numDynamicObstacles);
for i = 1:numDynamicObstacles
    dynamicObstacleHandles(i) = plot(dynamicObstaclePositions(i,2), dynamicObstaclePositions(i,1), 's', ...
        'MarkerSize', 10, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
end

% Set up the plot
axis equal;
axis([0.5 gridSize+0.5 0.5 gridSize+0.5]);
title('SupplyChainBot Simulation');
xlabel('X');
ylabel('Y');

% Create legend entries with both robot number and algorithm
legendEntries = cell(1, numRobots);
for i = 1:numRobots
    legendEntries{i} = sprintf('Robot %d (%s)', i, robotAlgorithms{i});
end
legend([robotHandles, supplyPointHandle, deliveryPointHandle, dynamicObstacleHandles(1)], ...
    [legendEntries, {'Supply Point', 'Delivery Point', 'Dynamic Obstacle'}], 'Location', 'eastoutside');

% Add path handles for each robot
pathHandles = gobjects(1, numRobots);

% Initialize metrics
metrics = struct();
for i = 1:numRobots
    metrics.pathLengths(i) = 0;
    metrics.computationTimes(i) = 0;
    metrics.deliveriesCompleted(i) = 0;
    metrics.distanceTraveled(i) = 0;
end

% Initialize delivery tracking matrix
deliveryMatrix = zeros(numDeliveryPoints, 1);

% Initialize supply tracking matrix
supplyMatrix = zeros(numRobots, 1);

% Main simulation loop
maxSteps = 1000;
step = 0;

[rows, cols] = size(environment);

while sum(deliveryMatrix) < numDeliveryPoints && step < maxSteps
    step = step + 1;
    
    % Update dynamic obstacles
    for i = 1:numDynamicObstacles
        newPosition = round(dynamicObstaclePositions(i,:) + dynamicObstacleVelocities(i,:));
        
        if newPosition(1) < 1 || newPosition(1) > gridSize
            dynamicObstacleVelocities(i,1) = -dynamicObstacleVelocities(i,1);
            newPosition(1) = max(1, min(gridSize, newPosition(1)));
        end
        if newPosition(2) < 1 || newPosition(2) > gridSize
            dynamicObstacleVelocities(i,2) = -dynamicObstacleVelocities(i,2);
            newPosition(2) = max(1, min(gridSize, newPosition(2)));
        end
        
        dynamicObstaclePositions(i,:) = newPosition;
        set(dynamicObstacleHandles(i), 'XData', newPosition(2), 'YData', newPosition(1));
    end
    
    % Robot movement loop
    for i = 1:numRobots
        % Determine the current goal (supply point or delivery point)
        if supplyMatrix(i) == 0
            distances = zeros(numSupplyPoints, 1);
            for v = 1:numSupplyPoints
                distances(v) = norm(supplyPointPositions(v,:) - robotPositions(i,:));
            end
            [~, nearestIdx] = min(distances);
            goal = supplyPointPositions(nearestIdx,:);
            isSupplyGoal = true;
        else
            undeliveredPoints = find(deliveryMatrix == 0);
            
            if ~isempty(undeliveredPoints)
                distances = zeros(length(undeliveredPoints), 1);
                for v = 1:length(undeliveredPoints)
                    deliveryIdx = undeliveredPoints(v);
                    distances(v) = norm(deliveryPointPositions(deliveryIdx,:) - robotPositions(i,:));
                end
                [~, nearestIdx] = min(distances);
                targetDeliveryIdx = undeliveredPoints(nearestIdx);
                goal = deliveryPointPositions(targetDeliveryIdx,:);
                isSupplyGoal = false;
            else
                continue;
            end
        end
        
        % Path planning
        tic;
        switch lower(robotAlgorithms{i})
            case 'a*'
                path = astar(round(robotPositions(i,:)), round(goal), environment);
            case 'dijkstra'
                path = dijkstra(round(robotPositions(i,:)), round(goal), environment);
            case 'bfs'
                path = bfs(round(robotPositions(i,:)), round(goal), environment);
            case 'genetic hybrid'
    path = genetic_algorithm_hybrid(round(robotPositions(i,:)), round(goal), environment, @(start, goal, env) astar(start, goal, env));
        end
        computationTime = toc;
        metrics.computationTimes(i) = metrics.computationTimes(i) + computationTime;
        
        % Visualize and follow path
        if ~isempty(path)
            if isvalid(pathHandles(i))
                delete(pathHandles(i));
            end
            pathHandles(i) = plot(path(:,2), path(:,1), ':', 'Color', robotColors{i}, 'LineWidth', 2);
            
            if size(path, 1) > 1
                newPosition = path(2,:);
                
                collisionDetected = false;
                for o = 1:numDynamicObstacles
                    if norm(newPosition - dynamicObstaclePositions(o,:)) < 1.5
                        collisionDetected = true;
                        break;
                    end
                end
                
                if ~collisionDetected && environment(newPosition(1), newPosition(2)) == 1
                    oldPosition = robotPositions(i,:);
                    robotPositions(i,:) = newPosition;
                    set(robotHandles(i), 'XData', newPosition(2), 'YData', newPosition(1));
                    
                    metrics.distanceTraveled(i) = metrics.distanceTraveled(i) + norm(newPosition - oldPosition);
                end
            end
            
            if isequal(round(robotPositions(i,:)), round(goal))
                if isSupplyGoal
                    supplyMatrix(i) = 1;
                else
                    if deliveryMatrix(targetDeliveryIdx) == 0
                        deliveryMatrix(targetDeliveryIdx) = 1;
                        supplyMatrix(i) = 0;
                        metrics.deliveriesCompleted(i) = metrics.deliveriesCompleted(i) + 1;
                        
                        remainingDeliveryPoints = find(deliveryMatrix == 0);
                        if ~isempty(remainingDeliveryPoints)
                            set(deliveryPointHandle, 'XData', deliveryPointPositions(remainingDeliveryPoints,2), ...
                                'YData', deliveryPointPositions(remainingDeliveryPoints,1));
                        else
                            set(deliveryPointHandle, 'XData', [], 'YData', []);
                        end
                    end
                end
            end
        end
    end
    
    drawnow;
    pause(0.1);
end

% Display final metrics
disp('Final Performance Metrics:');
for i = 1:numRobots
    fprintf('Robot %d (%s):\n', i, robotAlgorithms{i});
    fprintf('  Deliveries Completed: %d\n', metrics.deliveriesCompleted(i));
    fprintf('  Distance Traveled: %.2f\n', metrics.distanceTraveled(i));
    fprintf('  Total Computation Time: %.4f seconds\n', metrics.computationTimes(i));
    fprintf('\n');
end

% Clean up
for i = 1:numRobots
    if isvalid(pathHandles(i))
        delete(pathHandles(i));
    end
end