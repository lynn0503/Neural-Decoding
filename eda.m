clc;clear;
data=load('go_ins_dir.mat').go_ins_dir;
[trials,neurons]=size(data);
histogram(data(:,end))
%% firing rate distribution
% power law fitting
h1=figure(1);
vec=reshape(data,trials*neurons,1);
histfit(vec,30,'exponential')
title('firing rate histogram')
%% individual response over all trials
h2=figure(2);
for i=1:9
    subplot(3,3,i)
    y=movmean(data(:,i),5);
    plot(1:trials,data(:,i))
    xlabel('trials');ylabel('firing rate')
    title(['Neuron ',num2str(i)])
end

%% firing stability 
% i.e. individual response over specific direction 
neuron_id=12;
% direction=4;
for direction=1:8
    trial_idx=find(data(:,end)==direction);
    response=data(trial_idx,neuron_id);
    subplot(2,4,direction)
    histogram(response)
    title(['Direction ',num2str(direction)])
end
%% firing rate—direction
% neuron_id=12;
% direction=4;
fr_dir=zeros(4,1);
for nn=7:10
    for direction=1:8
        trial_idx=find(data(:,end)==direction);
        response=data(trial_idx,nn);
        fr_dir(direction)=mean(response);
    %     subplot(2,4,direction)
    %     histogram(response)
    %     title(['Direction ',num2str(direction)])
    end
subplot(2,2,nn-6)
plot(1:8,fr_dir,'linewidth',3,'color','k');
xlabel('direction');ylabel('mean firing rate');
title(['Neuron ',num2str(nn)])

end
%% feature correlation
corr_matrix=corr(data(:,1:end-1));
imagesc(corr_matrix)