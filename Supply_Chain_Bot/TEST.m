% Main script to run the Genetic Algorithm for pathfinding
clc;
clear;
close all;

% Define the environment size (grid size)
gridSize = [20, 20];  % Grid size (rows x columns)
obstacles = [5, 5; 7, 7; 10, 10; 15, 15];  % Example obstacles [row, col]
goalPos = [20, 20];  % Goal position
supplyPoint = [1, 1];  % Start position

% Create the environment grid
environment = ones(gridSize);  % Grid filled with 1s (free space)
for i = 1:size(obstacles, 1)
    environment(obstacles(i, 1), obstacles(i, 2)) = 0;  % 0 represents obstacles
end

% Run the Genetic Algorithm (Hybrid GA)
populationSize = 50;  % Number of individuals in the population
numGenerations = 100;  % Number of generations
mutationRate = 0.05;  % Mutation rate
numDynamicObstacles = 2;  % Example number of dynamic obstacles

% Initial population
population = cell(populationSize, 1);
for i = 1:populationSize
    population{i} = create_random_path(supplyPoint, goalPos, gridSize);  % Create random paths
end

% Main loop for Genetic Algorithm
for generation = 1:numGenerations
    fitness = zeros(populationSize, 1);
    
    % Evaluate fitness for each path in the population
    for i = 1:populationSize
        fitness(i) = evaluate_path(population{i}, environment, goalPos);
    end
    
    % Selection (Roulette Wheel or Tournament Selection can be used)
    selectedParents = selection(population, fitness);
    
    % Crossover (Create new offspring by crossover)
    offspring = crossover(selectedParents);
    
    % Mutation (Introduce small random changes to offspring)
    offspring = mutation(offspring, mutationRate, gridSize);
    
    % Create the new population
    population = offspring;
    
    % Optionally, display progress (show best path so far)
    bestPath = population{find(fitness == max(fitness), 1)};
    plot_path(bestPath, environment);  % Plot the best path
    pause(0.1);  % Pause to update the plot
end

% Final path result
finalPath = population{find(fitness == max(fitness), 1)};
disp('Best Path Found:');
disp(finalPath);

% Function to create a random path
function path = create_random_path(startPos, goalPos, gridSize)
    path = [startPos];
    currentPos = startPos;
    while ~isequal(currentPos, goalPos)
        possibleMoves = [0, 1; 1, 0; 0, -1; -1, 0];  % Up, Right, Down, Left
        move = possibleMoves(randi(4), :);
        nextPos = currentPos + move;
        if nextPos(1) >= 1 && nextPos(1) <= gridSize(1) && nextPos(2) >= 1 && nextPos(2) <= gridSize(2)
            currentPos = nextPos;
            path = [path; currentPos];
        end
    end
end

% Function to evaluate the path
function fitness = evaluate_path(path, environment, goalPos)
    % Ensure that all path coordinates are within the bounds of the environment
    path = round(path);  % Ensure the coordinates are integers
    path(path(:,1) < 1, :) = 1; % Ensure row indices are within bounds
    path(path(:,2) < 1, :) = 1; % Ensure column indices are within bounds
    path(path(:,1) > size(environment, 1), :) = size(environment, 1); % Check row bounds
    path(path(:,2) > size(environment, 2), :) = size(environment, 2); % Check column bounds
    
    % Check for collisions along the path (0 indicates an obstacle in the environment)
    collisions = sum(environment(sub2ind(size(environment), path(:,1), path(:,2))) == 0);
    
    % Compute the fitness: fewer collisions are better
    fitness = length(path) - collisions;  % Longer paths with fewer collisions are better
end

% Function for selection (Roulette Wheel or Tournament Selection)
function selectedParents = selection(population, fitness)
    % Simple Roulette Wheel Selection (can be replaced with Tournament Selection)
    totalFitness = sum(fitness);
    probabilities = fitness / totalFitness;
    cumulativeProbabilities = cumsum(probabilities);
    
    selectedParents = cell(length(population), 1);
    for i = 1:length(population)
        r = rand;
        selectedParents{i} = population{find(cumulativeProbabilities >= r, 1)};
    end
end

% Function for crossover (Single-point crossover)
function offspring = crossover(parents)
    offspring = cell(length(parents), 1);
    
    for i = 1:2:length(parents)-1
        % Ensure crossoverPoint does not exceed the length of the paths
        crossoverPoint = randi(min(length(parents{i}), length(parents{i+1})) - 1);
        
        % Create offspring by combining portions of both parents
        offspring{i} = [parents{i}(1:crossoverPoint, :); parents{i+1}(crossoverPoint+1:end, :)];
        offspring{i+1} = [parents{i+1}(1:crossoverPoint, :); parents{i}(crossoverPoint+1:end, :)];
    end
end

% Function for mutation (Random path adjustment)
function offspring = mutation(offspring, mutationRate, gridSize)
    for i = 1:length(offspring)
        if rand < mutationRate
            mutationPoint = randi(length(offspring{i}));
            mutationMove = randi(4);
            possibleMoves = [0, 1; 1, 0; 0, -1; -1, 0];
            offspring{i}(mutationPoint, :) = offspring{i}(mutationPoint, :) + possibleMoves(mutationMove, :);
            % Ensure the new position is within bounds
            offspring{i}(mutationPoint, :) = max(1, min(offspring{i}(mutationPoint, :), gridSize));
        end
    end
end

% Function to plot the path on the grid
function plot_path(path, environment)
    figure;
    imagesc(environment);
    hold on;
    plot(path(:,2), path(:,1), 'ro-', 'MarkerFaceColor','r');  % Plot the path
    title('Best Path Found');
    xlabel('Columns');
    ylabel('Rows');
    axis equal;
    axis tight;
    colorbar;
    hold off;
end
