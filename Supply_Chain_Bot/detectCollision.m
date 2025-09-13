function collision = detectCollision(position, environment, robotPositions)
    % Check if the position is within the environment bounds
    [rows, cols] = size(environment);
    if position(1) < 1 || position(1) > rows || position(2) < 1 || position(2) > cols
        collision = true; % Out of bounds collision
        return;
    end
    
    % Check if the position is an obstacle (e.g., walls, barriers)
    if environment(position(1), position(2)) == 0
        collision = true; % Obstacle collision
        return;
    end
    
    % Check if the position collides with other supply chain robots
    for i = 1:size(robotPositions, 1)
        if all(position == robotPositions(i,:))
            collision = true; % Robot collision
            return;
        end
    end
    
    % If we've made it here, there's no collision
    collision = false;
end
