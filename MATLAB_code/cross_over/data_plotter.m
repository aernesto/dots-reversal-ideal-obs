clear all
correct = load('H_tenth/correct.mat');
correct = correct.correct; % has size (1, nb subjects x nb trial)

time_elapsed = load('H_tenth/time_elapsed.mat');
time_elapsed = time_elapsed.time_elapsed;

time_elapsed_2 = load('H_tenth/time_elapsed_2.mat');
time_elapsed_2 = time_elapsed_2.time_elapsed_2;

BINS = 3;
%round up change point time to closest bin. 
% the wider the time bins, the more trials fall in each one --> higher 
% statistical power
%Ex. 4 bins make the bins 250 millisecond wide
%Ex. 10 bins make the bins 100 millisecond wide
%See line below for math to bin
time_elapsed_round = ceil(time_elapsed * BINS) / BINS;
X_1 = unique(time_elapsed_round); % only consider bin right end points for 
                                  % X-axis of plot
Y_1_count = zeros(2,size(X_1,2));
time_elapsed_2_round = ceil(time_elapsed_2 * BINS) / BINS;
X_1_bis = unique(time_elapsed_2_round); % only consider bin right end points for 
                                  % X-axis of plot
Y_1_count_bis = zeros(2,size(X_1_bis,2));
for i=1:size(correct, 2)  % loop over trials (all subjects confounded)
    
    ind = find(X_1==time_elapsed_round(i));
    Y_1_count(1,ind) = Y_1_count(1,ind) + correct(i);
    Y_1_count(2,ind) = Y_1_count(2,ind) + 1;
    ind_bis = find(X_1_bis==time_elapsed_2_round(i));
    Y_1_count_bis(1,ind_bis) = Y_1_count_bis(1,ind_bis) + correct(i);
    Y_1_count_bis(2,ind_bis) = Y_1_count_bis(2,ind_bis) + 1;
end

for i=1:size(Y_1_count,2)
   Y_1(1,i) = Y_1_count(1,i) / Y_1_count(2,i); % percentage correct
end
for i=1:size(Y_1_count_bis,2)
   Y_1_bis(1,i) = Y_1_count_bis(1,i) / Y_1_count_bis(2,i); % percentage correct
end

%compute standard deviation
err_1 = zeros(1,size(Y_1_count,2));
for i=1:size(Y_1_count,2)
    err_1(i) = std(repelem([0 1],[Y_1_count(1,i) (Y_1_count(2,i) - Y_1_count(1,i))]));
end

correct = load('H_2/correct.mat');
correct = correct.correct;

time_elapsed = load('H_2/time_elapsed.mat');
time_elapsed = time_elapsed.time_elapsed;

time_elapsed_2 = load('H_2/time_elapsed_2.mat');
time_elapsed_2 = time_elapsed_2.time_elapsed_2;

%round up to closest 

time_elapsed_round = ceil(time_elapsed * BINS) / BINS;
X_2 = unique(time_elapsed_round);
Y_2_count = zeros(2,size(X_2,2));

time_elapsed_2_round = ceil(time_elapsed_2 * BINS) / BINS;
X_2_bis = unique(time_elapsed_2_round);
Y_2_count_bis = zeros(2,size(X_2_bis,2));

for i=1:size(correct, 2)
    ind = find(X_2==time_elapsed_round(i));
    Y_2_count(1,ind) = Y_2_count(1,ind) + correct(i);
    Y_2_count(2,ind) = Y_2_count(2,ind) + 1;
    ind_bis = find(X_2_bis==time_elapsed_2_round(i));
    Y_2_count_bis(1,ind_bis) = Y_2_count_bis(1,ind_bis) + correct(i);
    Y_2_count_bis(2,ind_bis) = Y_2_count_bis(2,ind_bis) + 1;
end

for i=1:size(Y_2_count,2)
   Y_2(1,i) = Y_2_count(1,i) / Y_2_count(2,i); 
end

for i=1:size(Y_2_count_bis,2)
   Y_2_bis(1,i) = Y_2_count_bis(1,i) / Y_2_count_bis(2,i);
    
end

%compute standard deviation
err_2 = zeros(1,size(Y_2_count,2));
for i=1:size(Y_2_count,2)
    err_2(i) = std(repelem([0 1],[Y_2_count(1,i) (Y_2_count(2,i) - Y_2_count(1,i))]));
end

%plot everything
figure(1)
hold on
plot(X_1, Y_1(1,:),'Color',[0 0.4470 0.7410], 'Linewidth',3)
plot(X_2,Y_2(1,:), 'Color', [0.8500 0.3250 0.0980], 'Linewidth',3)
hold off
legend('Hazard = .1','Hazard  = 2')
xlabel('Time after Changepoint (seconds)')
ylabel('P(Correct)')
xlim([min(X_1(1), X_2(1)) max(X_1(end), X_2(end))])
ylim([0,1])
Y_1_count = cat(1,Y_1_count, X_1);
%plot everything
figure(2)
hold on
plot(X_1_bis, Y_1_bis(1,:),'Color',[0 0.4470 0.7410], 'Linewidth',3)
plot(X_2_bis,Y_2_bis(1,:), 'Color', [0.8500 0.3250 0.0980], 'Linewidth',3)
hold off
legend('Hazard = .1','Hazard  = 2')
xlabel('Time after Changepoint (seconds)')
ylabel('P(Correct)')
xlim([min(X_1_bis(1), X_2_bis(1)) max(X_1_bis(end), X_2_bis(end))])
ylim([0,1])
Y_1_count_bis = cat(1,Y_1_count_bis, X_1_bis);
