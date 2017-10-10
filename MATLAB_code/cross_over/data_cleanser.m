function [] = data_cleanser(file_name, H)
%%%%%%%%%%%
% ARGUMENTS:
%%%%%%%%%%%
% file_name: cell array containing file names, 
%            1 file per subject and per hazard rate 
% H        : one of two strings, '2' or 'tenth', corresponding to the 
%            true stimulus hazard rate
%%%%%%%%%%%
%   RETURNS:
%%%%%%%%%%%
all_data = [];                              % will be a vector of structs
for i=1:length(file_name)
    all_data = [all_data, load(file_name{i})]; 
end

[~, dim1] = size(all_data);                 % number of subjects
[dim2, ~] = size(all_data(1).statusData);   % number of trials

% we will store the time since the last change point and the subject's
% correctness for each trial and each subject
time_elapsed = zeros(1,dim1*dim2);
time_elapsed_2 = time_elapsed; % just another way of computing previous vec
correct= zeros(1,dim1 *dim2);

for i=1:dim1         % loop over subjects
   for j=1:dim2      % loop over trials
       directionvc = all_data(i).statusData(j).directionvc; % vector of 
                                                    % direction of motion
                                                    % 1 entry per frame 
       %find index of last changepoint
       tind = length(directionvc);
       last_changepoint_index = nan;
       for k=(tind - 1):-1:1
            if (directionvc(tind) ~= directionvc(k))
                last_changepoint_index = k;
                break;
            end
       end
       
       index = i*j;         % represents specific pair (subject i, trial j)
       
       %no changepoint,
       if (isnan(last_changepoint_index))
           %find the time since the last changepoint
           time_elapsed_2(index) = nan;
       else
           %find accuracy
           correct(index) = all_data(i).statusData(j).correct;
       
           %find the time since the last changepoint
           time_elapsed_2(index) = all_data(i).statusData(j).FST;
       end
   end
end

%remove trials that do not have changepoint
ind_remove = find(isnan(time_elapsed_2)); % returns indices of trials with
                                          % no change points
time_elapsed_2(ind_remove) = [];
correct(ind_remove) = [];

save(strcat('H_', H,'/correct'),'correct');
save(strcat('H_', H,'/time_elapsed'),'time_elapsed');
save(strcat('H_',H,'/time_elapsed_2'),'time_elapsed_2');
end