function path = genetic_algorithm_hybrid(startPos, goalPos, environment, otherAlgorithm)
    startPos = round(startPos);
    goalPos = round(goalPos);
    [rows, cols] = size(environment);

    % Increase population size and reduce generations for faster computation
    populationSize = 50;
    mutationRate = 0.1;
    numGenerations = 30;
    eliteCount = 5;

    % Initialize population with a mix of A* and intelligent random paths
    population = cell(populationSize, 1);
    for i = 1:populationSize
        if i <= eliteCount
            population{i} = otherAlgorithm(startPos, goalPos, environment);
        else
            population{i} = intelligent_random_path(startPos, goalPos, environment);
        end
    end

    bestFitness = -inf;
    bestPath = [];

    for gen = 1:numGenerations
        % Evaluate fitness
        fitness = zeros(populationSize, 1);
        for i = 1:populationSize
            fitness(i) = evaluate_path(population{i}, environment, goalPos);
            if fitness(i) > bestFitness
                bestFitness = fitness(i);
                bestPath = population{i};
            end
        end

        % Early termination if perfect path found
        if any(fitness == inf)
            path = population{find(fitness == inf, 1)};
            return;
        end

        % Select parents
        [~, sortedIndices] = sort(fitness, 'descend');
        elites = population(sortedIndices(1:eliteCount));
        parents = population(sortedIndices(1:floor(populationSize/2)));

        % Create new population
        newPopulation = cell(populationSize, 1);
        newPopulation(1:eliteCount) = elites;  % Elitism
        
        for i = eliteCount+1:populationSize
            if rand < 0.8  % 80% chance of crossover
                parent1 = parents{randi(length(parents))};
                parent2 = parents{randi(length(parents))};
                child = intelligent_crossover(parent1, parent2, environment);
            else
                child = intelligent_random_path(startPos, goalPos, environment);
            end
            child = mutate(child, mutationRate, environment);
            newPopulation{i} = child;
        end

        population = newPopulation;
    end

    % If no perfect path found, return the best path
    path = bestPath;
end

function fitness = evaluate_path(path, environment, goalPos)
    if isempty(path)
        fitness = -inf;
        return;
    end
    pathLength = size(path, 1);
    collisions = sum(environment(sub2ind(size(environment), path(:,1), path(:,2))) == 0);
    distanceToGoal = norm(path(end,:) - goalPos);
    
    if collisions == 0 && all(path(end,:) == goalPos)
        fitness = inf;  % Perfect path
    else
        fitness = 1000 / (pathLength + 1000*collisions + distanceToGoal);
    end
end

function child = intelligent_crossover(parent1, parent2, environment)
    [rows, cols] = size(environment);
    crossoverPoint = randi([1, min(size(parent1,1), size(parent2,1))]);
    child = [parent1(1:crossoverPoint,:); parent2(crossoverPoint+1:end,:)];
    
    % Remove loops
    [~, uniqueIndices] = unique(child, 'rows', 'stable');
    child = child(uniqueIndices,:);
    
    % Ensure path validity
    for i = 2:size(child, 1)
        while norm(child(i,:) - child(i-1,:)) > sqrt(2)
            intermediatePoint = round((child(i,:) + child(i-1,:)) / 2);
            intermediatePoint = max(min(intermediatePoint, [rows, cols]), [1, 1]);
            child = [child(1:i-1,:); intermediatePoint; child(i:end,:)];
        end
    end
end

function path = mutate(path, mutationRate, environment)
    [rows, cols] = size(environment);
    for i = 2:size(path, 1)-1  % Don't mutate start and end points
        if rand < mutationRate
            newPoint = path(i,:) + randi([-1,1], 1, 2);
            newPoint = max(min(newPoint, [rows, cols]), [1, 1]);
            if environment(newPoint(1), newPoint(2)) == 1
                path(i,:) = newPoint;
            end
        end
    end
    
    % Remove any resulting loops
    [~, uniqueIndices] = unique(path, 'rows', 'stable');
    path = path(uniqueIndices,:);
end

function path = intelligent_random_path(startPos, goalPos, environment)
    [rows, cols] = size(environment);
    path = startPos;
    currentPos = startPos;
    maxSteps = 200;  % Increased max steps
    
    for i = 1:maxSteps
        diff = goalPos - currentPos;
        probabilities = abs(diff) / sum(abs(diff));
        
        if rand < 0.8  % 80% chance to move towards the goal
            if rand < probabilities(1)
                step = [sign(diff(1)), 0];
            else
                step = [0, sign(diff(2))];
            end
        else  % 20% chance for a random move
            step = randsample([-1, 0, 1], 2, true);
        end
        
        newPos = currentPos + step;
        newPos = max(min(newPos, [rows, cols]), [1, 1]);
        
        if environment(newPos(1), newPos(2)) == 1
            path = [path; newPos];
            currentPos = newPos;
            if all(currentPos == goalPos)
                break;
            end
        end
    end
    
    if ~all(path(end,:) == goalPos)
        path = [path; goalPos];
    end
    
    % Optimize path
    path = optimize_path(path, environment);
end

function path = optimize_path(path, environment)
    i = 1;
    while i < size(path, 1) - 1
        for j = size(path, 1):-1:i+2
            if can_connect_directly(path(i,:), path(j,:), environment)
                path = [path(1:i,:); path(j:end,:)];
                break;
            end
        end
        i = i + 1;
    end
end

function canConnect = can_connect_directly(start, goal, environment)
    diff = goal - start;
    distance = max(abs(diff));
    if distance == 0
        canConnect = true;
        return;
    end
    
    steps = diff / distance;
    points = round(start + (0:distance)' * steps);
    canConnect = all(environment(sub2ind(size(environment), points(:,1), points(:,2))));
end