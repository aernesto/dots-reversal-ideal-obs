clear
clc
% Compute the performance of the ideal-observer model for a single point in
% parameter space.
%
% ARGUMENTS: 
% The 5 components of the vector params are
%   SNR = positive scalar   Signal-to-Noise Ratio
%   T = integer             interrogation time
%   h = between 0 and 1     hazard rate
%   alpha, beta positive scalars    hyperparams for the Beta prior over h
% In addition:
%   nSims = integer         Number of independent simulations over which
%                           calculate performance
% OUTPUT:
%   perf is a scalar between 0 and 1

% Expand params vector into specific variables:
SNR=2;
T=4;
h=0.1;
alpha=1;
beta=1;
nSims = 1;
timePoints=floor(linspace(T,T)); %points for which sims are run
testOBS = [.7,-.2,-2,3.6];

%frequency of correct answers at each time point for known rate algorithms
m = 1;                      %half the distance between means of likelihoods

sigma=2*m/SNR;         %sd of likelihoods

priorPrec=alpha+beta;

%start loop on time points
c=1;
N=timePoints(c);    %time at which each simulation stops.
freqU=0;              %frequency of correct responses for unknown rate

%loop over sims
for s=1:nSims
%initialize variables
Hpn = zeros(N,1); Hpc = Hpn; Hmn = Hpn; Hmc = Hpn;
Pp = zeros(T);
Pm = Pp;
%generate first state randomly
    if rand<.5
        strue=m;
    else
        strue=-m;
    end
%make an observation
x = testOBS(1);

%algorithm for unknown rate - first time step
%compute joint probabilities over state and change point count
        Hpc(1) = exp(-(x-m)^2/(2*sigma^2));
       % Hpc'
        Hmc(1) = exp(-(x+m)^2/(2*sigma^2));
        Fd = Hpc(1)+Hmc(1);
        Hpc(1) = Hpc(1)/Fd;
       % Hpc'
        Hmc(1) = Hmc(1)/Fd;
        %compute marginals over state
        lp = Hpc(1);
        lm = Hmc(1);
        Pp(:,1) = Hpc;
        Pm(:,1) = Hmc;
    %pursue the algorithms if the interrogation time is >1     
    %loop over time
    for j=1:N-1
        %update the true state
        if rand<h
            strue=-strue;
        end

        % make an observation
        x = testOBS(j+1);
        %compute likelihoods
        prefactor = 1 / sqrt(2*pi*sigma^2);
        xp = exp(-(x-m)^2/(2*sigma^2))*prefactor;
        xm = exp(-(x+m)^2/(2*sigma^2))*prefactor;

        %specifics of unknown rate algorithm
        % update the boundaries (with 0 and j changepoints)
                ea = 1-alpha/(j-1+priorPrec);
                eb = (j-1+alpha)/(j-1+priorPrec);
                Hpn(1) = xp*ea*Hpc(1);
                Hmn(1) = xm*ea*Hmc(1);
                Hpn(j+1) = xp*eb*Hmc(j);
                Hmn(j+1) = xm*eb*Hpc(j);
%                 if j > N-2
%                     Hpn'
%                     Hmn'
%                 end
                
                % update the interior values
                if j>1
                    vk = (2:j)';
                    ep = 1-(vk-1+alpha)/(j-1+priorPrec);   %no change
                    em=(vk-2+alpha)/(j-1+priorPrec);       %change
                    Hpn(vk) = xp*(ep.*Hpc(vk)+em.*Hmc(vk-1));
                    Hmn(vk) = xm*(ep.*Hmc(vk)+em.*Hpc(vk-1));
                end
                
                % sum probabilities in order to normalize
                Hs = sum(Hpn)+sum(Hmn);
                Hpc=Hpn/Hs;
                Hmc=Hmn/Hs;
                Pp(:,j+1)=Hpc;
                Pm(:,j+1)=Hmc;
                %compute marginals over state if last iteration
%                 if j==N-1
%                     lp = sum(Hpc); lm = sum(Hmc);
%                     j
%                     vk
%                     ea
%                     eb
%                     ep
%                     em
%                     Hpc
%                     Hmc
                %end           
    end

    %compute decisions (interrogate the system)
            decvaru=m*sign(log(lp/lm));
            if decvaru==strue
                freqU=freqU+1;           %count correct answers
            end
end
     perf(c)=freqU;
     perf=perf/nSims;
     Pp
     Pm