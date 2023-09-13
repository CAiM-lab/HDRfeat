function out = LDRtoHDR_agc(input, expo)

input = Clamp(input, 0, 1);
gray_input = mat2gray(input);
mean_input = mean(gray_input, 'all');
gamma = log(0.5)/log(mean_input);
gamma
% input = single(input);
out = (input).^gamma;
out = out ./ expo;
