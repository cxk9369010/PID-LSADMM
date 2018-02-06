
clc
clear
close all
%% LASSO data which extracted from simulationTable1.m
tempY=[31,54,110,76,99;
    37,63,138,95,123;
    45,72,165,113,146;
    54,81,193,132,170;
    62,91,221,150,194];



    
    
%     subplot(1,2,i)
    figure(1);
    b=bar(tempY);
    hold on;
    color=[0 0 0.75;
        0 0.75 0;
        0 0.75 0.75;
        0.75 0 0;
        1 1 0];
    for j=1:5
        set(b(j),'FaceColor',color(j,:));
    end
    
    xlabel('Tolerances($\epsilon_1$,$\epsilon_2$)','Interpreter','latex','fontsize',12);
    ylabel('Iteration Numbers')
    legend('LSADMM-1-2','GS-ADMM-III','PJALM','TADMM','HTY')
    set(gca,'XTickLabel',{'1e-6,1e-7','1e-7,1e-8','1e-8,1e-9','1e-9,1e-10','1e-10,1e-11'})


  

