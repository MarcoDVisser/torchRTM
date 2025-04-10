% 定义输入变量
load('coef_data.mat');

load('angles_data.mat');

%angles.tts = 30;
%angles.tto = 10;
%angles.psi = 45;

atm.alt_m = 500;        % 海拔高度
atm.Pa0 = 0;         % 初始气压（假设是一个特殊值）-999
atm.aot550 = 0.325;       % 气溶胶光学厚度
atm.uo3 = 0.35;          % 臭氧含量
atm.Pa = 1013.25%[];  % Leave empty if you want to calculate it inside SMAC

atm.uh2o = 1.41;           % 水汽含量

% 调用SMAC函数并打印结果
Opt_atm = SMAC(angles, atm,coef);
disp(Opt_atm);
% 保存结果为.mat文件
save('Opt_atm_result.mat', 'Opt_atm', '-v7');

%when using python:

%scipy.io.loadmat('Opt_atm_result.mat') #ValueError: Unknown mat file type, version 10, 32
