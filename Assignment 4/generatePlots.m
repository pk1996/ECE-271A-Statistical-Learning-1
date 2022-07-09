clear;
xlabel = cell(10,1);
d = [1,2,4,8,16,32,40,48,56,64];
for i = 1 : 10
    xlabel{i} = d(i);
end

for i = 1 : 25
    name = strcat('GMM_', int2str(i));
    fName = strcat(name, '_6a', '.mat');
    load(fName);
    plot(pError,'o-');
    
    hold on;
    if (mod(i,5) == 0)
        title(strcat('GMM #', int2str(i/5), ' for foreground'))
        f = gcf;
        ax = f.Children;
        ax.XTickLabel = xlabel;
        legend('GMM_BG_1','GMM_BG_2','GMM_BG_3','GMM_BG_4','GMM_BG_5')
        f.Units = "normalized";
        f.Position = [0 0 1 1];
        for idx = 1:5
            ax.Children(idx).LineWidth = 2;
        end
        ax.XLabel.String = 'Dimension of Feature Space';
        ax.YLabel.String = 'Probability of Error';

        fName = strcat('Q6(a)_', int2str(i/5), '.png');
        saveas(f, fName)
        close all;
    end
end
close all;

for i = 1 : 6
    name = strcat('GMM_', int2str(d(i)));
    fName = strcat(name, '_Q6b', '.mat');
    load(fName);
    plot(pError,'o-');
    hold on;
end


title('GMM with different number of components')
f = gcf;
ax = f.Children;
ax.XTickLabel = xlabel;
legend('1', '2', '4', '8', '16', '32')
f.Units = "normalized";
f.Position = [0 0 1 1];
for idx = 1:6
    ax.Children(idx).LineWidth = 2;
end
ax.XLabel.String = 'Dimension of Feature Space';
ax.YLabel.String = 'Probability of Error';

fName = strcat('Q6(b)', '.png');
saveas(f, fName)
close all;