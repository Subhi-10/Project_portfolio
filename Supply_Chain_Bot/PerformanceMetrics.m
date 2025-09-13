classdef PerformanceMetrics < handle
    properties
        pathLengths         % Total path lengths for each robot
        computationTimes    % Total computation times for each robot
        deliveriesMade      % Number of deliveries made by each robot
        distanceTraveled    % Total distance traveled by each robot
        algorithmNames      % Names of algorithms used for routing
    end
    
    methods
        function obj = PerformanceMetrics(numRobots, algorithmNames)
            % Initialize performance metrics for each robot
            obj.pathLengths = zeros(numRobots, 1);         % Path lengths for each robot
            obj.computationTimes = zeros(numRobots, 1);    % Computation times for each robot
            obj.deliveriesMade = zeros(numRobots, 1);      % Deliveries made by each robot
            obj.distanceTraveled = zeros(numRobots, 1);     % Distance traveled by each robot
            obj.algorithmNames = algorithmNames;            % Names of algorithms
        end
        
        function updatePathMetrics(obj, robotIndex, pathLength, computationTime)
            % Update path length and computation time for a specific robot
            obj.pathLengths(robotIndex) = obj.pathLengths(robotIndex) + pathLength;
            obj.computationTimes(robotIndex) = obj.computationTimes(robotIndex) + computationTime;
        end
        
        function updateDeliveryMetrics(obj, robotIndex, distance)
            % Update delivery metrics for a specific robot
            obj.deliveriesMade(robotIndex) = obj.deliveriesMade(robotIndex) + 1;
            obj.distanceTraveled(robotIndex) = obj.distanceTraveled(robotIndex) + distance;
        end
        
        function displayMetrics(obj)
            % Display the performance metrics in a bar graph format
            figure('Name', 'Supply Chain Performance Metrics', 'Position', [100, 100, 800, 600]);
            
            % Path Lengths
            subplot(2, 2, 1);
            bar(obj.pathLengths);
            title('Total Path Length');
            ylabel('Length (units)');
            set(gca, 'XTickLabel', obj.algorithmNames);
            
            % Computation Times
            subplot(2, 2, 2);
            bar(obj.computationTimes);
            title('Total Computation Time');
            ylabel('Time (seconds)');
            set(gca, 'XTickLabel', obj.algorithmNames);
            
            % Deliveries Made
            subplot(2, 2, 3);
            bar(obj.deliveriesMade);
            title('Deliveries Made');
            ylabel('Count');
            set(gca, 'XTickLabel', obj.algorithmNames);
            
            % Distance Traveled
            subplot(2, 2, 4);
            bar(obj.distanceTraveled);
            title('Distance Traveled');
            ylabel('Distance (units)');
            set(gca, 'XTickLabel', obj.algorithmNames);
        end
    end
end
