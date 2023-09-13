function InitParam()

global param;
global gamma;
global mu;

gamma = 2.2;
mu = 5000;

%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.trainingScenes = '/mnt/asgard2/data/lingkai/SIGGRAPH17_HDR_Trainingset/Training';
param.trainingData = 'Result/Training/';

param.testScenes = '/mnt/asgard2/data/lingkai/SIGGRAPH17_HDR_Trainingset/Test/Main';
param.testData = 'Result/Test/';


param.cropSizeTraining = 10; % we crop the boundaries to avoid artifacts in the training
param.border = 6;












