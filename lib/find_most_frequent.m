function n = find_most_frequent(tab,edges)
%FIND_MOST_FREQUENT Summary of this function goes here
%   Detailed explanation goes here
freq = histcounts(tab(tab>0),edges);
n = find(freq==max(freq))+1;
end

