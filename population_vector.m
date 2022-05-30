function y_predicted = population_vector(data,feature)

if feature==1
    cols=1:142;
elseif feature==2
    cols=143:284;
elseif feature==3
    cols=1:284;
end

x_train=data.x_train(:,cols);
x_test=data.x_test(:,cols);

max_dir=train(x_train,data.y_train);
y_predicted=predict(x_test,max_dir);

    function max_dir=train(x_train,y_train)
        [~, num_neurons]=size(x_train);
        mean_fr=zeros(8,num_neurons);
       
        for i=1:8
            mean_fr(i,:)=mean(x_train(y_train==i,:));
        end

        %% fit cosine function
        para_init=[0 1 0];
        para=zeros(num_neurons,3);
        max_dir=zeros(num_neurons,2);
        deg_discrete=0:pi/4:(2*pi-pi/4);
        deg_continous=0:0.001:2*pi;
        mean_fr_pv=zeros(8,num_neurons);
        for i=1:num_neurons
            para(i,:)=nlinfit(deg_discrete,mean_fr(:,i),@fitting_function,para_init);
            fitted=fitting_function(para(i,:),deg_continous);
            [max_fr, index]=max(fitted);
            max_dir(i,:)=[max_fr,deg_continous(index)];
%             for j=1:8
%                 mean_fr_pv(j,i)=fitted(round(1/j*length(fitted)));
%             end
        end
        save trained_pv.mat para max_dir
        
        function y=fitting_function(p,x)
            y=p(1)+p(2)*cos(x'-p(3));
        end

    end

    function y_predicted=predict(x_test,max_dir)
        [num_trials,~]=size(x_test);
        pop_vec_x=max_dir(:,1).*cos(max_dir(:,2)).*x_test';
        pop_vec_y=max_dir(:,1).*sin(max_dir(:,2)).*x_test';
        deg=mod(atan2(sum(pop_vec_y),sum(pop_vec_x))/pi*180,360);
        %bin direction into one of eight targets
        dir = zeros(num_trials,1);
        dir(deg >337.5 | deg <=22.5)=1;
        deg_binned=0:45:315;
        for i = 2:8
            dir(deg >deg_binned(i)-22.5 & deg <=deg_binned(i)+22.5) = i;
        end
        y_predicted=dir;
    end

end

