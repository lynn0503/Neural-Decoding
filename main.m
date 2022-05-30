%% 
clc;clear;
%% get data
load('go_ins_dir.mat');
%% predict
% feature: 1 for go, 2 for instruction, 3 for both
%% pv
% acc_pv=zeros(1,3);
% data_pv=split_data(go_ins_dir,0.8);
% for i=1:3
%     y_predicted=population_vector(data_pv,i);
%     acc_pv(i)=score_acc(y_predicted,data_pv.y_test);
% end
%% nb
k=100;
ACC=zeros(k,4);
for j=1:k
    disp(['processing ',num2str(j),'th loop'])
    data=split_data(go_ins_dir,0.8);
    
%     disp('processing population vector')
    y_predicted_pv=population_vector(data,1);
    ACC(j,1)=score_acc(y_predicted_pv,data.y_test);
%     disp('processing gaussian naive bayes')
    y_predicted_nb_normal=naive_bayes(data,'gausse',1,0);
    ACC(j,2)=score_acc(y_predicted_nb_normal',data.y_test);
%     disp('processing poisson naive bayes')
    y_predicted_nb_poison=naive_bayes(data,'poison',1,0);
    ACC(j,3)=score_acc(y_predicted_nb_poison',data.y_test);
%     disp('processing weighted poisson naive bayes')
    y_predicted_poison_weighted=naive_bayes(data,'poison',1,1);
    ACC(j,4)=score_acc(y_predicted_poison_weighted',data.y_test);
end
ACC=mean(ACC)';
%% compare accuracy
bar(ACC,'FaceColor',[0.5 0.5 0.5]);
text(1:length(ACC), ACC, num2str(ACC,'%0.2f'),'FontSize', 16,'HorizontalAlignment','center','VerticalAlignment','bottom');
% add baseline
hold on 
plot(xlim,[0.125 0.125])
labels=["Population Vector";"Gaussian NB";"Poisson NB";"Weighted Poisson NB"];
set(gca,'xticklabel',labels,'YLim',[0,1])
title('Accuracy');
%% aux functions
function data=split_data(dataset_all,pct)
[rows,cols]=size(dataset_all);
% train set
train_size=round(rows*pct);
train_index=randperm(rows,train_size);
train_set=dataset_all(train_index,:);
x_train=train_set(:,1:cols-1);
y_train=train_set(:,end);
% test set
dataset_all(train_index,:)=[];
test_set=dataset_all;
x_test=test_set(:,1:cols-1);
y_test=test_set(:,end);
data=struct('x_train',x_train,'y_train',y_train,'x_test',x_test,'y_test',y_test);
save dataset.mat x_train y_train x_test y_test
end

% evaluate models by accuracy
function acc=score_acc(y_predicted,y_test)
 acc=sum(y_predicted==y_test)/length(y_predicted);
end

