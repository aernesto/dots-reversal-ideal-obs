counter = zeros(200,1);
for j = 1:200                           % loop over trials
    V = statusData(j).directionvc;      % get direction of dots on each frame
    duration = statusData(j).duration;  % get trial duration in seconds
    for i=1:(length(V)-1)               % loop over image frames (stimulus)
        if not(V(i+1)==V(i))            % change point presence
            counter(j) = counter(j) + 1;
        end
    end
    counter(j) = counter(j) / duration; % convert change point count to hazard rate 
                                        % estimate 
end
mean(counter)                           % average estimates across trials