function y_predicted = naive_bayes(data,distribution,feature,wgt)
% poisson naive bayes model

if feature==1
    cols=1:142;
elseif feature==2
    cols=143:284;
elseif feature==3
    cols=1:284;
end

x_train=data.x_train(:,cols);
% x_train=x_train-mean(x_train);
x_test=data.x_test(:,cols);
% x_test=x_test-mean(x_test);
% disp('training')
train(x_train,data.y_train,wgt);
% disp('predicting')
y_predicted=predict(x_test,distribution);

    
    function train(x_train,y_train,wgt)
        [num_trials, num_neurons]=size(x_train);
        mean_fr=zeros(8,num_neurons);
        std_fr=zeros(8,num_neurons);
        pct_dir=zeros(1,8);
        for i=1:8
            trials_i_dir=find(y_train==i);
            mean_fr(i,:)=mean(x_train(trials_i_dir,:));
            std_fr(i,:)=std(x_train(trials_i_dir,:));
            pct_dir(i)=length(trials_i_dir)/num_trials;
        end
%         pct_dir=zeros(1,8)+1/8;
        if wgt==1
            weights=weighting(x_train,y_train);
        else 
            weights=ones(1,num_neurons);
        end
        
        save trained_nb.mat mean_fr std_fr pct_dir weights
        
        function w=weighting(x,y)
            [rows,col]=size(x);
            corr_mat=corr(x);
            corr_xx=sum(abs(corr_mat));
%             mi_xx=zeros(col,col);
%             for ii=1:col
%                 for jj=(ii+1):col
%                     mi_xx(ii,jj)=MI(x(:,ii),x(:,jj));
%                 end
%             end
%             for ii=1:col
%                 for jj=1:ii
%                     if jj==ii
%                         mi_xx(ii,jj)=0;
%                     else 
%                         mi_xx(ii,jj)=mi_xx(jj,ii);
%                     end
%                 end
%             end
%             corr_xx=mean(mi_xx);
            
            mi_xy=ones(1,col);
            for j=1:col                
                mi_xy(j)=MI(x(:,j),y);
            end
            w=mi_xy./corr_xx;
            
            function mi=MI(v1,v2)
                len=length(v1);
                [x_cnt,x_unique]=groupcounts(v1);
                p_x=zeros(1,max(x_unique)+1);% take 0 into account
                p_x(x_unique+1)=x_cnt/len;
         
                [y_cnt,y_unique]=groupcounts(v2);
                p_y=zeros(1,max(y_unique)+1);
                p_y(y_unique+1)=y_cnt/len;

                xy=[v1 v2];
                [xy_unique,~,ic]=unique(xy,'rows');
                xy_cnt=accumarray(ic,1);
                p_xy=xy_cnt/len;
                
                mi=sum(p_xy.*log(p_xy./(p_x(xy_unique(:,1)+1).*p_y(xy_unique(:,2)+1))),'all');
                
            end    
        end
    end

    function y_predicted=predict(x_test,distribution)
        stats=load('trained_nb.mat');
        mean_fr=stats.mean_fr;
        std_fr=stats.std_fr;
        pct_dir=stats.pct_dir;
        weights=stats.weights;
        
%         mean_fr_pv=load('mean_fr_pv.mat');
%         mean_fr=mean_fr_pv.mean_fr_pv;
        [num_trials,~]=size(x_test);
        prob_apost_log=zeros(8,num_trials);
        for t=1:num_trials
            for d=1:8
                if distribution=='poison'
                    prob_likeli=poisspdf(x_test(t,:),mean_fr(d,:));
                else
                    prob_likeli=normpdf(x_test(t,:),mean_fr(d,:),std_fr(d,:));
                end
                prob_aprior=pct_dir(d);
%                 prob_apost_log(d,t)=sum(log([prob_likeli prob_aprior]));% no weight
                prob_apost_log(d,t)=sum(log([prob_likeli.^weights prob_aprior]));
            end
        end
        [~,dir]=max(prob_apost_log);
        y_predicted=dir;
    end

end

