classdef SupplyChainPriorityQueue < handle
    properties (Access = private)
        elements   % Cell array to store elements (nodes)
        priorities % Array to store the associated priorities (costs or distances)
    end
    
    methods
        function obj = SupplyChainPriorityQueue()
            obj.elements = {};        % Initialize empty cell array for elements
            obj.priorities = [];      % Initialize empty array for priorities
        end
        
        % Insert an element with its associated priority
        function insert(obj, element, priority)
            obj.elements{end+1} = element;  % Add the new element
            obj.priorities(end+1) = priority; % Set the associated priority
        end
        
        % Remove and return the element with the highest priority (lowest cost)
        function element = pop(obj)
            [~, idx] = min(obj.priorities); % Find index of the element with the lowest priority
            element = obj.elements{idx};     % Retrieve the element
            obj.elements(idx) = [];           % Remove the element from the queue
            obj.priorities(idx) = [];         % Remove the associated priority
        end
        
        % Check if the queue is empty
        function result = isEmpty(obj)
            result = isempty(obj.elements); % Return true if there are no elements
        end
        
        % Check if the queue contains a specific element
        function result = contains(obj, element)
            result = any(cellfun(@(x) isequal(x, element), obj.elements)); % Return true if the element is found
        end
        
        % Decrease the priority of an element if the new priority is lower
        function decreaseKey(obj, element, newPriority)
            idx = find(cellfun(@(x) isequal(x, element), obj.elements), 1); % Find index of the element
            if ~isempty(idx) && newPriority < obj.priorities(idx) % Check if the new priority is lower
                obj.priorities(idx) = newPriority; % Update the priority
            end
        end
    end
end
