% File: PriorityQueue.m
classdef PriorityQueue < handle
    properties
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
        
        function tf = isEmpty(obj)
            tf = isempty(obj.elements);
        end
        
        function tf = contains(obj, element)
            tf = any(cellfun(@(x) isequal(x, element), obj.elements));
        end
        
        function decrease_key(obj, element, new_priority)
            idx = find(cellfun(@(x) isequal(x, element), obj.elements), 1);
            if ~isempty(idx)
                obj.priorities(idx) = new_priority;
            end
        end
        
        function s = size(obj)
            s = length(obj.elements);
        end
    end
end