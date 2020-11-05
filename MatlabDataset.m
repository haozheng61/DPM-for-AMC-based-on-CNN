%%% This code is copied from
%%% https://www.mathworks.com/help/deeplearning/ug/modulation-classification-with-deep-learning.html?s_tid=srchtitle"
%%% Please run it on MATLAB R2020a.
%%% Do not use the random seed.
%%% Before running this code, excute
%       "openExample('deeplearning_shared/ModulationClassificationWithDeepLearningExample')"
%       in the MATLAB Command Window to get the filepath of the relative codes, and put this code into
%       that filepath

clear
for SNR=[-6:4:30]
    modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
      "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", ...
      "B-FM", "DSB-AM" ,"SSB-AM"]);
    numFramesPerModType = 400;
    sps = 8;                % Samples per symbol
    spf = 1024;             % Samples per frame
    symbolsPerFrame = spf / sps;
    fs = 200e3;             % Sample rate
    fc = [902e6 100e6];     % Center frequencies
    maxDeltaOff = 5;
    deltaOff = (rand()*2*maxDeltaOff) - maxDeltaOff;
    C = 1 + (deltaOff/1e6);
    channel = helperModClassTestChannel(...   % Open the source code and replace the "Rician" with "Rayleigh" while simulating the Rayleigh fading channel 
      'SampleRate', fs, ...
      'SNR', SNR, ...
      'PathDelays', [0 1.8 3.4] / fs, ...
      'AveragePathGains', [0 -2 -10], ...
      'KFactor', 4, ...   % Delete this while simulating the Rayleigh fading channel 
      'MaximumDopplerShift', 5, ...   % 5/ 50/ 100/ 200
      'MaximumClockOffset', 5, ...
      'CenterFrequency', 902e6);
    %rng(1235)   % Please remember to delete this random seed
    tic
    numModulationTypes = length(modulationTypes);
    channelInfo = info(channel);
    transDelay = 50;
    fileNameRoot = "frame";
    for modType = 1:numModulationTypes
        fprintf('%s - Generating %s frames\n', ...
          datestr(toc/86400,'HH:MM:SS'), modulationTypes(modType))
        label = modulationTypes(modType);
        numSymbols = (numFramesPerModType / sps);
        dataSrc = helperModClassGetSource(modulationTypes(modType), sps, 2*spf, fs);
        modulator = helperModClassGetModulator(modulationTypes(modType), sps, fs);
        if contains(char(modulationTypes(modType)), {'B-FM','DSB-AM','SSB-AM'})
          % Analog modulation types use a center frequency of 100 MHz
          channel.CenterFrequency = 100e6;
        else
          % Digital modulation types use a center frequency of 902 MHz
          channel.CenterFrequency = 902e6;
        end
        frame_list=[];
        for p=1:numFramesPerModType
          % Generate random data
          x = dataSrc();

          % Modulate
          y = modulator(x);

          % Pass through independent channels
          rxSamples = channel(y);

          % Remove transients from the beginning, trim to size, and normalize
          frame1 = helperModClassFrameGenerator(rxSamples, spf, spf, transDelay, sps);
          frame = frame1(769:1024);   % We select the last 256 sample points for AMC
          frame_list=[frame_list;frame'];
        end
        modulationTypes(modType)
        dir = strcat('D:\MATLAB\Examples\R2020a\deeplearning_shared\ModulationClassificationWithDeepLearningExample\datasets\',string(modulationTypes(modType)),num2str(SNR),'db.mat');
        save(dir,"frame_list")
     end
end
