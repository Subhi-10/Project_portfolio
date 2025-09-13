function path = project1(start, goal, map)
    [rows, cols] = size(map);
    
    % Create the open and closed sets
    openSet = PriorityQueue();
    closedSet = zeros(size(map));
    
    % Add the start node to the open set
    openSet.insert(start, 0);
    
    % Initialize the g_score and f_score maps
    g_score = inf(size(map));
    f_score = inf(size(map));
    g_score(start(1), start(2)) = 0;
    f_score(start(1), start(2)) = heuristic(start, goal);
    
    % Initialize the came_from map
    came_from = cell(size(map));
    
    % Define possible movements (up, down, left, right, diagonals)
    movements = [-1 0; 1 0; 0 -1; 0 1; -1 -1; -1 1; 1 -1; 1 1];
    
    while ~openSet.isEmpty()
        current = openSet.pop();
        
        if isequal(current, goal)
            path = reconstructPath(came_from, current);
            return;
        end
        
        closedSet(current(1), current(2)) = 1;
        
        for i = 1:size(movements, 1)
            neighbor = current + movements(i, :);
            
            % Check if the neighbor is within the map and not an obstacle
            if neighbor(1) > 0 && neighbor(1) <= rows && ...
               neighbor(2) > 0 && neighbor(2) <= cols && ...
               map(neighbor(1), neighbor(2)) ~= 0 && ...
               closedSet(neighbor(1), neighbor(2)) == 0
                
                tentative_g_score = g_score(current(1), current(2)) + 1;
                
                if tentative_g_score < g_score(neighbor(1), neighbor(2))
                    came_from{neighbor(1), neighbor(2)} = current;
                    g_score(neighbor(1), neighbor(2)) = tentative_g_score;
                    f_score(neighbor(1), neighbor(2)) = g_score(neighbor(1), neighbor(2)) + heuristic(neighbor, goal);
                    
                    if ~openSet.contains(neighbor)
                        openSet.insert(neighbor, f_score(neighbor(1), neighbor(2)));
                    else
                        openSet.decreaseKey(neighbor, f_score(neighbor(1), neighbor(2)));
                    end
                end
            end
        end
    end
    
    % If we get here, there's no path
    path = [];
end

function h = heuristic(a, b)
    % Manhattan distance
    h = abs(a(1) - b(1)) + abs(a(2) - b(2));
end

function path = reconstructPath(came_from, current)
    path = current;
    while ~isempty(came_from{current(1), current(2)})
        current = came_from{current(1), current(2)};
        path = [current; path];
    end
end

% Priority Queue implementation (simplified for this example)
classdef PriorityQueue < handle
    properties (Access = private)
        elements
        priorities
    end
    
    methods
        function obj = PriorityQueue()
            obj.elements = {};
            obj.priorities = [];
        end
        
        function insert(obj, element, priority)
            obj.elements{end+1} = element;
            obj.priorities(end+1) = priority;
        end
        
        function element = pop(obj)
            [~, idx] = min(obj.priorities);
            element = obj.elements{idx};
            obj.elements(idx) = [];
            obj.priorities(idx) = [];
        end
        
        function result = isEmpty(obj)
            result = isempty(obj.elements);
        end
        
        function result = contains(obj, element)
            result = any(cellfun(@(x) isequal(x, element), obj.elements));
        end
        
        function decreaseKey(obj, element, newPriority)
            idx = find(cellfun(@(x) isequal(x, element), obj.elements), 1);
            if ~isempty(idx)
                obj.priorities(idx) = newPriority;
            end
        end
    end
end
% Search and Rescue Simulation - A* Integration

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
robotHandles = gobjects(1, numRobots);
for i = 1:numRobots
    robotHandles(i) = plot(robotPositions(i,2), robotPositions(i,1), 'o', 'MarkerSize', 10, 'MarkerFaceColor', robotColors{i}, 'MarkerEdgeColor', 'k');
end

% Plot victims
victimHandles = plot(victimPositions(:,2), victimPositions(:,1), 'p', 'MarkerSize', 10, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k');

% Set up the plot
axis equal;
axis([0.5 gridSize+0.5 0.5 gridSize+0.5]);
title('Search and Rescue Simulation');
xlabel('X');
ylabel('Y');

legend([robotHandles, victimHandles(1)], {'Robot 1', 'Robot 2', 'Robot 3', 'Victim'}, 'Location', 'eastoutside');

% Main simulation loop
for step = 1:100 % Run for 100 steps
    for i = 1:numRobots
        % Find the nearest victim for each robot
        distances = sqrt(sum((victimPositions - robotPositions(i,:)).^2, 2));
        [~, nearestVictimIndex] = min(distances);
        goal = victimPositions(nearestVictimIndex, :);
        
        % Use A* to find a path to the nearest victim
        path = astar(robotPositions(i,:), goal, environment);
        
        % Move the robot along the path (one step at a time)
        if ~isempty(path) && size(path, 1) > 1
            robotPositions(i,:) = path(2,:);
            set(robotHandles(i), 'XData', robotPositions(i,2), 'YData', robotPositions(i,1));
        end
        
        % Check if the robot has reached a victim
        if isequal(robotPositions(i,:), goal)
            % Remove the victim
            victimPositions(nearestVictimIndex, :) = [];
            delete(victimHandles(nearestVictimIndex));
            victimHandles(nearestVictimIndex) = [];
            
            % Exit the simulation if all victims are rescued
            if isempty(victimPositions)
                disp('All victims rescued!');
                return;
            end
        end
    end
    
    % Update the plot
    drawnow;
    pause(0.1); % Add a small delay to make the animation visible
end

hold off;