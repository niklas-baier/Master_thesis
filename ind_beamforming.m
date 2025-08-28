% Batch Beamforming for ASR with Enhanced Speech Activity Detection
% This code processes multiple audio segments in a directory

clc; clear; close all;

%% Configuration
% Directory paths
inputDir = '/media/niklas/SSD2/ind_beamforming/';
outputDir = '/media/niklas/SSD2/final_beamforming/';

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Parameters
micSpacing = 0.035;     % 3.5 cm spacing between microphones
c = 343;                % Speed of sound (m/s)
channels = [1, 7, 4];   % Channel numbers to use

%% Find all audio segments
fprintf('Scanning directory for audio segments...\n');

% Get all wav files
wavFiles = dir(fullfile(inputDir, '*.wav'));
segmentGroups = {};

% Group files by segment identifier
for i = 1:length(wavFiles)
    filename = wavFiles(i).name;
    
    % Extract segment identifier (everything before the last underscore and number)
    % Example: S03_U01.CH1_1.wav -> S03_U01_1
    tokens = regexp(filename, '(.+)\.CH(\d+)_(\d+)\.wav', 'tokens');
    if ~isempty(tokens)
        baseId = tokens{1}{1};  % S03_U01
        segmentNum = tokens{1}{3};  % 1
        segmentId = [baseId '_' segmentNum];  % S03_U01_1
        
        if ~isfield(segmentGroups, segmentId) || isempty(segmentGroups.(segmentId))
            segmentGroups.(segmentId) = {};
        end
        segmentGroups.(segmentId){end+1} = filename;
    end
end

% Convert to cell array for easier processing
segmentIds = fieldnames(segmentGroups);
fprintf('Found %d audio segments to process\n', length(segmentIds));

%% Process each segment
successCount = 0;
failCount = 0;

for segIdx = 1:length(segmentIds)
    segmentId = segmentIds{segIdx};
    files = segmentGroups.(segmentId);
    
    fprintf('\n=== Processing segment %d/%d: %s ===\n', segIdx, length(segmentIds), segmentId);
    
    try
        % Find files for specified channels
        channelFiles = cell(length(channels), 1);
        channelFound = false(length(channels), 1);
        
        for chIdx = 1:length(channels)
            chNum = channels(chIdx);
            % Look for file with this channel
            for fileIdx = 1:length(files)
                if contains(files{fileIdx}, sprintf('.CH%d_', chNum))
                    channelFiles{chIdx} = fullfile(inputDir, files{fileIdx});
                    channelFound(chIdx) = true;
                    break;
                end
            end
        end
        
        if ~all(channelFound)
            fprintf('Warning: Missing channels for segment %s, skipping\n', segmentId);
            failCount = failCount + 1;
            continue;
        end
        
        % Process this segment
        outputFilename = strrep(files{find(channelFound, 1)}, '.wav', '.wav');
        outputPath = fullfile(outputDir, outputFilename);
        
        success = processAudioSegment(channelFiles, outputPath, micSpacing, c);
        
        if success
            successCount = successCount + 1;
            fprintf('Successfully processed: %s\n', outputFilename);
        else
            failCount = failCount + 1;
            fprintf('Failed to process: %s\n', segmentId);
        end
        
    catch ME
        fprintf('Error processing segment %s: %s\n', segmentId, ME.message);
        failCount = failCount + 1;
    end
end

fprintf('\n=== Processing Complete ===\n');
fprintf('Successfully processed: %d segments\n', successCount);
fprintf('Failed: %d segments\n', failCount);

%% Main processing function
function success = processAudioSegment(filePaths, outputPath, micSpacing, c)
    success = false;
    
    try
        nChannels = length(filePaths);
        
        %% Load Audio Data
        fprintf('  Loading %d channels...\n', nChannels);
        audioData = [];
        fs = [];
        
        for i = 1:nChannels
            [data, currentFs] = audioread(filePaths{i});
            
            if i == 1
                fs = currentFs;
                nSamples = length(data);
                audioData = zeros(nSamples, nChannels);
            else
                if currentFs ~= fs
                    error('Sampling rate mismatch between files.');
                end
                if length(data) ~= nSamples
                    minLength = min(nSamples, length(data));
                    audioData = audioData(1:minLength, :);
                    data = data(1:minLength);
                    nSamples = minLength;
                end
            end
            
            audioData(:, i) = data;
        end
        
        % Skip very short segments (less than 0.5 seconds)
        if nSamples < 0.5 * fs
            fprintf('  Segment too short (%d samples), skipping\n', nSamples);
            return;
        end
        
        %% Enhanced Speech Activity Detection
        fprintf('  Performing VAD...\n');
        speechMask = performEnhancedVAD(audioData(:, 1), fs);
        speechPercentage = 100 * sum(speechMask) / nSamples;
        fprintf('  Speech activity: %.1f%%\n', speechPercentage);
        
        %% Direction Detection (simplified for batch processing)
        fprintf('  Detecting optimal direction...\n');
        optimalDirection = detectOptimalDirection(audioData, speechMask, micSpacing, c, fs);
        fprintf('  Optimal direction: %.1f degrees\n', optimalDirection(1));
        
        %% Apply Beamforming
        fprintf('  Applying beamforming...\n');
        beamformedSignal = applyASROptimizedBeamforming(audioData, optimalDirection, speechMask, micSpacing, c, fs);
        
        %% Post-processing for ASR
        fprintf('  Post-processing for ASR...\n');
        beamformedSignal = postProcessForASR(beamformedSignal, fs);
        
        %% Save Output
        audiowrite(outputPath, beamformedSignal, fs);
        success = true;
        
    catch ME
        fprintf('  Error in processAudioSegment: %s\n', ME.message);
    end
end

%% Enhanced VAD function (simplified for batch processing)
function speechMask = performEnhancedVAD(signal, fs)
    frameSize = round(0.025 * fs);  % 25ms frames
    hopSize = round(0.01 * fs);     % 10ms hop
    nSamples = length(signal);
    nFrames = floor((nSamples - frameSize) / hopSize) + 1;
    
    energy = zeros(nFrames, 1);
    
    for i = 1:nFrames
        startIdx = (i-1) * hopSize + 1;
        endIdx = min(startIdx + frameSize - 1, nSamples);
        frame = signal(startIdx:endIdx);
        energy(i) = sum(frame.^2);
    end
    
    % Adaptive thresholding
    energyThreshold = prctile(energy, 55);  % More conservative for ASR
    speechFrames = energy > energyThreshold;
    
    % Apply median filtering
    speechFrames = medfilt1(double(speechFrames), 3) > 0.5;
    
    % Create speech mask
    speechMask = false(nSamples, 1);
    for i = 1:nFrames
        startIdx = (i-1) * hopSize + 1;
        endIdx = min(startIdx + frameSize - 1, nSamples);
        if speechFrames(i)
            speechMask(startIdx:endIdx) = true;
        end
    end
    
    % Ensure minimum speech detection
    if sum(speechMask) < 0.05 * nSamples
        energyThreshold = prctile(energy, 30);
        speechFrames = energy > energyThreshold;
        speechMask = false(nSamples, 1);
        for i = 1:nFrames
            startIdx = (i-1) * hopSize + 1;
            endIdx = min(startIdx + frameSize - 1, nSamples);
            if speechFrames(i)
                speechMask(startIdx:endIdx) = true;
            end
        end
    end
end

%% Optimal direction detection (simplified)
function optimalDirection = detectOptimalDirection(audioData, speechMask, micSpacing, c, fs)
    nChannels = size(audioData, 2);
    array = phased.ULA('NumElements', nChannels, 'ElementSpacing', micSpacing);
    
    % Use speech segments for direction detection
    speechIndices = find(speechMask);
    if length(speechIndices) < fs/2  % Need at least 0.5 seconds
        speechIndices = 1:length(speechMask);
    end
    
    % Sample for efficiency
    sampleSize = min(length(speechIndices), 2 * fs);  % Use up to 2 seconds
    stepSize = max(1, floor(length(speechIndices) / sampleSize));
    sampleIndices = speechIndices(1:stepSize:end);
    sampleIndices = sampleIndices(1:min(end, sampleSize));
    
    speechSample = audioData(sampleIndices, :);
    
    % Coarser scan for batch processing (10 degree steps)
    azimuthRange = -90:10:90;
    maxPower = -Inf;
    optimalDirection = [0; 0];
    
    for az = azimuthRange
        try
            scanBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
                'SampleRate', fs, 'PropagationSpeed', c, 'Direction', [az; 0]);
            
            beamOutput = scanBeamformer(speechSample);
            power = mean(beamOutput.^2);
            
            if power > maxPower
                maxPower = power;
                optimalDirection = [az; 0];
            end
        catch
            % Skip problematic directions
            continue;
        end
    end
    
    % Fine-tune with 2° resolution
    fineRange = (optimalDirection(1)-8):2:(optimalDirection(1)+8);
    fineRange = fineRange(fineRange >= -90 & fineRange <= 90);
    
    for az = fineRange
        try
            scanBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
                'SampleRate', fs, 'PropagationSpeed', c, 'Direction', [az; 0]);
            
            beamOutput = scanBeamformer(speechSample);
            power = mean(beamOutput.^2);
            
            if power > maxPower
                maxPower = power;
                optimalDirection = [az; 0];
            end
        catch
            continue;
        end
    end
end

%% ASR-Optimized Beamforming
function beamformedSignal = applyASROptimizedBeamforming(audioData, optimalDirection, speechMask, micSpacing, c, fs)
    nChannels = size(audioData, 2);
    array = phased.ULA('NumElements', nChannels, 'ElementSpacing', micSpacing);
    
    % Primary beamforming
    primaryBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
        'SampleRate', fs, 'PropagationSpeed', c, 'Direction', optimalDirection);
    
    beamformedSignal = primaryBeamformer(audioData);
    
    % Conservative spectral subtraction (less aggressive for ASR)
    beamformedSignal = applyConservativeSpectralSubtraction(beamformedSignal, fs, speechMask);
    
    % Normalize
    maxVal = max(abs(beamformedSignal));
    if maxVal > 0
        beamformedSignal = beamformedSignal / maxVal * 0.9;
    end
end

%% Conservative spectral subtraction for ASR
function enhancedSignal = applyConservativeSpectralSubtraction(signal, fs, speechMask)
    frameSize = round(0.032 * fs);
    hopSize = round(0.016 * fs);
    alpha = 1.2;  % Reduced over-subtraction for ASR
    beta = 0.2;   % Higher spectral floor for ASR
    
    noiseIndices = find(~speechMask);
    if isempty(noiseIndices)
        enhancedSignal = signal;
        return;
    end
    
    % Estimate noise spectrum
    noiseSegment = signal(noiseIndices(1:min(end, frameSize*5)));
    if length(noiseSegment) < frameSize
        enhancedSignal = signal;
        return;
    end
    
    noiseSpectrum = abs(fft(noiseSegment .* hann(length(noiseSegment))));
    noiseSpectrum = noiseSpectrum(1:floor(length(noiseSpectrum)/2)+1);
    
    % Process in frames
    nFrames = floor((length(signal) - frameSize) / hopSize) + 1;
    enhancedSignal = zeros(size(signal));
    window = hann(frameSize);
    
    for i = 1:nFrames
        startIdx = (i-1) * hopSize + 1;
        endIdx = startIdx + frameSize - 1;
        
        if endIdx > length(signal)
            break;
        end
        
        frame = signal(startIdx:endIdx) .* window;
        frameSpectrum = fft(frame);
        frameMagnitude = abs(frameSpectrum);
        framePhase = angle(frameSpectrum);
        
        enhancedMagnitude = frameMagnitude(1:floor(length(frameMagnitude)/2)+1);
        noiseEst = noiseSpectrum(1:length(enhancedMagnitude));
        
        % Conservative spectral subtraction
        enhancedMagnitude = enhancedMagnitude - alpha * noiseEst;
        enhancedMagnitude = max(enhancedMagnitude, beta * frameMagnitude(1:length(enhancedMagnitude)));
        
        enhancedSpectrum = [enhancedMagnitude; flipud(enhancedMagnitude(2:end-1))];
        enhancedSpectrum = enhancedSpectrum .* exp(1j * framePhase);
        
        enhancedFrame = real(ifft(enhancedSpectrum)) .* window;
        enhancedSignal(startIdx:endIdx) = enhancedSignal(startIdx:endIdx) + enhancedFrame;
    end
end

%% ASR-specific post-processing
function processedSignal = postProcessForASR(signal, fs)
    % High-pass filter (more conservative for ASR)
    [b1, a1] = butter(3, 60/(fs/2), 'high');  % 60 Hz high-pass
    processedSignal = filtfilt(b1, a1, signal);
    
    % Speech band emphasis (wider band for ASR robustness)
    [b2, a2] = butter(2, [100 7000]/(fs/2), 'bandpass');
    speechEnhanced = filtfilt(b2, a2, processedSignal);
    
    % Conservative combination
    processedSignal = 0.9 * processedSignal + 0.1 * speechEnhanced;
    
    % Final normalization with headroom for ASR
    maxVal = max(abs(processedSignal));
    if maxVal > 0
        processedSignal = processedSignal / maxVal * 0.8;  % More headroom for ASR
    end
end