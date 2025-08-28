% Whisper-Optimized Batch Beamforming for ASR
% Tailored specifically for Whisper model input with MFCC-aware processing

clc; clear; close all;

%% Configuration
% Directory paths
inputDir = '/media/niklas/SSD2/ind_beamforming/';
outputDir = '/media/niklas/SSD2/whisper_beamforming/';

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Whisper-specific parameters
micSpacing = 0.035;     % 3.5 cm spacing between microphones
c = 343;                % Speed of sound (m/s)
channels = [1, 7, 4];   % Channel numbers to use
targetFs = 16000;       % Whisper's expected sample rate

% MFCC analysis parameters for evaluation
mfccParams.numCoeffs = 13;
mfccParams.frameSize = round(0.025 * targetFs);  % 25ms
mfccParams.hopSize = round(0.01 * targetFs);     % 10ms
mfccParams.numFilters = 26;

%% Find all audio segments
fprintf('Scanning directory for audio segments...\n');

% Get all wav files
wavFiles = dir(fullfile(inputDir, '*.wav'));
segmentGroups = {};

% Group files by segment identifier
for i = 1:length(wavFiles)
    filename = wavFiles(i).name;
    
    % Extract segment identifier
    tokens = regexp(filename, '(.+)\.CH(\d+)_(\d+)\.wav', 'tokens');
    if ~isempty(tokens)
        baseId = tokens{1}{1};
        segmentNum = tokens{1}{3};
        segmentId = [baseId '_' segmentNum];
        
        if ~isfield(segmentGroups, segmentId) || isempty(segmentGroups.(segmentId))
            segmentGroups.(segmentId) = {};
        end
        segmentGroups.(segmentId){end+1} = filename;
    end
end

segmentIds = fieldnames(segmentGroups);
fprintf('Found %d audio segments to process\n', length(segmentIds));

%% Process each segment with MFCC analysis
successCount = 0;
failCount = 0;
mfccResults = struct();

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
        
        % Process this segment with MFCC analysis
        outputFilename = strrep(files{find(channelFound, 1)}, '.wav', '_whisper.wav');
        outputPath = fullfile(outputDir, outputFilename);
        
        [success, mfccMetrics] = processAudioSegmentForWhisper(channelFiles, outputPath, ...
            micSpacing, c, targetFs, mfccParams);
        
        if success
            successCount = successCount + 1;
            mfccResults.(segmentId) = mfccMetrics;
            fprintf('Successfully processed: %s\n', outputFilename);
            fprintf('  MFCC SNR improvement: %.2f dB\n', mfccMetrics.snrImprovement);
            fprintf('  Spectral distortion: %.3f\n', mfccMetrics.spectralDistortion);
        else
            failCount = failCount + 1;
            fprintf('Failed to process: %s\n', segmentId);
        end
        
    catch ME
        fprintf('Error processing segment %s: %s\n', segmentId, ME.message);
        failCount = failCount + 1;
    end
end

% Save MFCC analysis results
save(fullfile(outputDir, 'mfcc_analysis_results.mat'), 'mfccResults');

fprintf('\n=== Processing Complete ===\n');
fprintf('Successfully processed: %d segments\n', successCount);
fprintf('Failed: %d segments\n', failCount);

if successCount > 0
    avgSnrImprovement = mean(arrayfun(@(x) x.snrImprovement, struct2array(mfccResults)));
    avgSpectralDistortion = mean(arrayfun(@(x) x.spectralDistortion, struct2array(mfccResults)));
    fprintf('Average MFCC SNR improvement: %.2f dB\n', avgSnrImprovement);
    fprintf('Average spectral distortion: %.3f\n', avgSpectralDistortion);
end

%% Main processing function optimized for Whisper
function [success, mfccMetrics] = processAudioSegmentForWhisper(filePaths, outputPath, ...
    micSpacing, c, targetFs, mfccParams)
    
    success = false;
    mfccMetrics = struct();
    
    try
        nChannels = length(filePaths);
        
        %% Load and preprocess audio data
        fprintf('  Loading %d channels...\n', nChannels);
        audioData = [];
        originalFs = [];
        
        for i = 1:nChannels
            [data, currentFs] = audioread(filePaths{i});
            
            if i == 1
                originalFs = currentFs;
                % Resample to Whisper's target sample rate
                if currentFs ~= targetFs
                    data = resample(data, targetFs, currentFs);
                end
                nSamples = length(data);
                audioData = zeros(nSamples, nChannels);
            else
                if currentFs ~= originalFs
                    error('Sampling rate mismatch between files.');
                end
                % Resample to target rate
                if currentFs ~= targetFs
                    data = resample(data, targetFs, currentFs);
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
        
        % Skip very short segments
        if nSamples < 0.5 * targetFs
            fprintf('  Segment too short (%d samples), skipping\n', nSamples);
            return;
        end
        
        %% Compute baseline MFCCs for comparison
        baselineMfccs = computeMFCCs(audioData(:, 1), targetFs, mfccParams);
        
        %% Enhanced Speech Activity Detection (Whisper-aware)
        fprintf('  Performing Whisper-aware VAD...\n');
        speechMask = performWhisperAwareVAD(audioData(:, 1), targetFs);
        speechPercentage = 100 * sum(speechMask) / nSamples;
        fprintf('  Speech activity: %.1f%%\n', speechPercentage);
        
        %% Whisper-optimized direction detection
        fprintf('  Detecting optimal direction...\n');
        optimalDirection = detectOptimalDirectionForWhisper(audioData, speechMask, ...
            micSpacing, c, targetFs);
        fprintf('  Optimal direction: %.1f degrees\n', optimalDirection(1));
        
        %% Apply Whisper-optimized beamforming
        fprintf('  Applying Whisper-optimized beamforming...\n');
        beamformedSignal = applyWhisperOptimizedBeamforming(audioData, optimalDirection, ...
            speechMask, micSpacing, c, targetFs);
        
        %% Minimal post-processing for Whisper
        fprintf('  Minimal post-processing for Whisper...\n');
        beamformedSignal = minimalPostProcessForWhisper(beamformedSignal, targetFs);
        
        %% Compute enhanced MFCCs and metrics
        enhancedMfccs = computeMFCCs(beamformedSignal, targetFs, mfccParams);
        mfccMetrics = analyzeMFCCImpact(baselineMfccs, enhancedMfccs, speechMask, mfccParams);
        
        %% Save output
        audiowrite(outputPath, beamformedSignal, targetFs);
        success = true;
        
    catch ME
        fprintf('  Error in processAudioSegmentForWhisper: %s\n', ME.message);
    end
end

%% Whisper-aware VAD (preserves more speech content)
function speechMask = performWhisperAwareVAD(signal, fs)
    % Whisper is robust to some noise, so be more inclusive
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
    
    % Adaptive thresholding (more conservative for Whisper)
    energyThreshold = prctile(energy, 45);  % Lower threshold than original
    speechFrames = energy > energyThreshold;
    
    % Apply median filtering
    speechFrames = medfilt1(double(speechFrames), 3) > 0.5;
    
    % Create speech mask - same logic as original
    speechMask = false(nSamples, 1);
    for i = 1:nFrames
        startIdx = (i-1) * hopSize + 1;
        endIdx = min(startIdx + frameSize - 1, nSamples);
        if speechFrames(i)
            speechMask(startIdx:endIdx) = true;
        end
    end
    
    % Ensure minimum speech detection (same as original logic)
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

%% Whisper-optimized direction detection
function optimalDirection = detectOptimalDirectionForWhisper(audioData, speechMask, ...
    micSpacing, c, fs)
    
    nChannels = size(audioData, 2);
    array = phased.ULA('NumElements', nChannels, 'ElementSpacing', micSpacing);
    
    % Use more speech content for direction detection
    speechIndices = find(speechMask);
    if length(speechIndices) < fs/4  % Need at least 0.25 seconds
        speechIndices = 1:length(speechMask);
    end
    
    % Use more data for better estimation
    sampleSize = min(length(speechIndices), 4 * fs);  % Up to 4 seconds
    stepSize = max(1, floor(length(speechIndices) / sampleSize));
    sampleIndices = speechIndices(1:stepSize:end);
    sampleIndices = sampleIndices(1:min(end, sampleSize));
    
    speechSample = audioData(sampleIndices, :);
    
    % Optimize for speech clarity rather than just power
    azimuthRange = -90:5:90;  % 5-degree resolution
    maxMetric = -Inf;
    optimalDirection = [0; 0];
    
    for az = azimuthRange
        try
            scanBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
                'SampleRate', fs, 'PropagationSpeed', c, 'Direction', [az; 0]);
            
            beamOutput = scanBeamformer(speechSample);
            
            % Compute speech-optimized metric
            metric = computeSpeechQualityMetric(beamOutput, fs);
            
            if metric > maxMetric
                maxMetric = metric;
                optimalDirection = [az; 0];
            end
        catch
            continue;
        end
    end
    
    % Fine-tune with 1° resolution
    fineRange = (optimalDirection(1)-4):1:(optimalDirection(1)+4);
    fineRange = fineRange(fineRange >= -90 & fineRange <= 90);
    
    for az = fineRange
        try
            scanBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
                'SampleRate', fs, 'PropagationSpeed', c, 'Direction', [az; 0]);
            
            beamOutput = scanBeamformer(speechSample);
            metric = computeSpeechQualityMetric(beamOutput, fs);
            
            if metric > maxMetric
                maxMetric = metric;
                optimalDirection = [az; 0];
            end
        catch
            continue;
        end
    end
end

%% Speech quality metric for direction optimization
function metric = computeSpeechQualityMetric(signal, fs)
    % Combine power and spectral characteristics
    power = mean(signal.^2);
    
    % Spectral characteristics favorable for speech
    [psd, f] = pwelch(signal, [], [], [], fs);
    speechBand = (f >= 300) & (f <= 3400);  % Traditional speech band
    speechPower = sum(psd(speechBand));
    totalPower = sum(psd);
    
    spectralRatio = speechPower / totalPower;
    
    % Combine metrics
    metric = power * (1 + spectralRatio);
end

%% Whisper-optimized beamforming (minimal distortion)
function beamformedSignal = applyWhisperOptimizedBeamforming(audioData, optimalDirection, ...
    speechMask, micSpacing, c, fs)
    
    nChannels = size(audioData, 2);
    array = phased.ULA('NumElements', nChannels, 'ElementSpacing', micSpacing);
    
    % Primary beamforming with minimal distortion
    primaryBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
        'SampleRate', fs, 'PropagationSpeed', c, 'Direction', optimalDirection);
    
    beamformedSignal = primaryBeamformer(audioData);
    
    % NO spectral subtraction - Whisper handles noise well
    % Only apply gentle noise suppression if SNR is very poor
    snrEstimate = estimateSNR(beamformedSignal, speechMask);
    
    if snrEstimate < -5  % Only for very noisy signals
        beamformedSignal = applyGentleNoiseReduction(beamformedSignal, speechMask, fs);
    end
    
    % Normalize conservatively
    maxVal = max(abs(beamformedSignal));
    if maxVal > 0
        beamformedSignal = beamformedSignal / maxVal * 0.95;  % High headroom
    end
end

%% Gentle noise reduction (only for very noisy cases)
function enhancedSignal = applyGentleNoiseReduction(signal, speechMask, fs)
    % Very conservative Wiener filtering
    frameSize = round(0.032 * fs);
    hopSize = round(0.016 * fs);
    
    noiseIndices = find(~speechMask);
    if isempty(noiseIndices) || length(noiseIndices) < frameSize
        enhancedSignal = signal;
        return;
    end
    
    % Estimate noise PSD
    noiseSegment = signal(noiseIndices(1:min(end, frameSize*3)));
    noisePSD = pwelch(noiseSegment, hann(frameSize), [], frameSize, fs);
    
    % Process in frames with gentle Wiener filtering
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
        [signalPSD, f] = pwelch(frame, [], [], [], fs);
        
        % Conservative Wiener gain
        wienerGain = signalPSD ./ (signalPSD + 0.5 * noisePSD);  % Conservative
        wienerGain = max(wienerGain, 0.3);  % High noise floor
        
        % Apply filtering in frequency domain
        frameSpectrum = fft(frame);
        enhancedSpectrum = frameSpectrum;
        
        % Only apply to positive frequencies
        nFreqs = length(wienerGain);
        enhancedSpectrum(1:nFreqs) = enhancedSpectrum(1:nFreqs) .* wienerGain;
        if mod(length(frameSpectrum), 2) == 0
            enhancedSpectrum(end-nFreqs+2:end) = enhancedSpectrum(end-nFreqs+2:end) .* flipud(wienerGain(2:end-1));
        else
            enhancedSpectrum(end-nFreqs+2:end) = enhancedSpectrum(end-nFreqs+2:end) .* flipud(wienerGain(2:end));
        end
        
        enhancedFrame = real(ifft(enhancedSpectrum)) .* window;
        enhancedSignal(startIdx:endIdx) = enhancedSignal(startIdx:endIdx) + enhancedFrame;
    end
end

%% Minimal post-processing for Whisper
function processedSignal = minimalPostProcessForWhisper(signal, fs)
    % Whisper expects natural audio - minimal processing
    
    % Only remove DC offset and very low frequencies
    [b, a] = butter(2, 50/(fs/2), 'high');  % 50 Hz high-pass
    processedSignal = filtfilt(b, a, signal);
    
    % NO bandpass filtering - Whisper uses full spectrum
    
    % Final normalization with generous headroom
    maxVal = max(abs(processedSignal));
    if maxVal > 0
        processedSignal = processedSignal / maxVal * 0.9;
    end
end

%% MFCC computation
function mfccs = computeMFCCs(signal, fs, params)
    % Compute MFCCs for analysis
    frameSize = params.frameSize;
    hopSize = params.hopSize;
    numCoeffs = params.numCoeffs;
    
    try
        % Use MATLAB's mfcc function with valid parameters
        mfccs = mfcc(signal, fs, 'WindowLength', frameSize, ...
                     'OverlapLength', frameSize - hopSize, ...
                     'NumCoeffs', numCoeffs, ...
                     'LogEnergy', 'Ignore');
        
        % Transpose to match expected format (coeffs x frames)
        mfccs = mfccs';
        
    catch ME
        % Fallback: manual MFCC computation if built-in function fails
        fprintf('    Warning: Using fallback MFCC computation\n');
        mfccs = computeMFCCsManual(signal, fs, params);
    end
end

%% Manual MFCC computation (fallback)
function mfccs = computeMFCCsManual(signal, fs, params)
    frameSize = params.frameSize;
    hopSize = params.hopSize;
    numCoeffs = params.numCoeffs;
    numFilters = params.numFilters;
    
    % Ensure signal is column vector
    if size(signal, 2) > size(signal, 1)
        signal = signal';
    end
    
    % Frame the signal
    nFrames = max(1, floor((length(signal) - frameSize) / hopSize) + 1);
    
    % Mel filter bank parameters
    lowFreq = 80;  % Start from 80 Hz for speech
    highFreq = min(fs/2, 8000);  % Cap at 8kHz for speech
    melLow = 2595 * log10(1 + lowFreq/700);
    melHigh = 2595 * log10(1 + highFreq/700);
    melPoints = linspace(melLow, melHigh, numFilters + 2);
    hzPoints = 700 * (10.^(melPoints/2595) - 1);
    
    % Create mel filter bank
    fftSize = 2^nextpow2(frameSize);  % Use power of 2 for FFT efficiency
    nFreqBins = floor(fftSize/2) + 1;
    filterBank = zeros(numFilters, nFreqBins);
    
    for i = 1:numFilters
        left = max(1, round(hzPoints(i) * fftSize / fs) + 1);
        center = max(1, round(hzPoints(i+1) * fftSize / fs) + 1);
        right = max(1, round(hzPoints(i+2) * fftSize / fs) + 1);
        
        % Ensure indices are within bounds
        left = min(left, nFreqBins);
        center = min(center, nFreqBins);
        right = min(right, nFreqBins);
        
        % Skip if filter would be too narrow
        if right <= left || center <= left || right <= center
            continue;
        end
        
        % Rising slope
        if center > left
            for j = left:center
                filterBank(i, j) = (j - left) / (center - left);
            end
        end
        
        % Falling slope
        if right > center
            for j = center:right
                filterBank(i, j) = (right - j) / (right - center);
            end
        end
    end
    
    % Compute MFCCs
    mfccs = zeros(numCoeffs, nFrames);
    window = hann(frameSize);
    
    for i = 1:nFrames
        startIdx = (i-1) * hopSize + 1;
        endIdx = min(startIdx + frameSize - 1, length(signal));
        
        % Extract frame with zero padding if needed
        if endIdx > length(signal)
            frame = [signal(startIdx:end); zeros(frameSize - (length(signal) - startIdx + 1), 1)];
        elseif endIdx - startIdx + 1 < frameSize
            frame = [signal(startIdx:endIdx); zeros(frameSize - (endIdx - startIdx + 1), 1)];
        else
            frame = signal(startIdx:endIdx);
        end
        
        % Apply window and FFT
        windowed = frame .* window;
        spectrum = fft(windowed, fftSize);
        powerSpectrum = abs(spectrum(1:nFreqBins)).^2;
        
        % Apply mel filter bank
        melEnergy = filterBank * powerSpectrum;
        melEnergy = max(melEnergy, eps); % Avoid log(0)
        
        % Log and DCT
        logMelEnergy = log(melEnergy);
        dctCoeffs = dct(logMelEnergy);
        
        % Extract requested number of coefficients
        actualCoeffs = min(numCoeffs, length(dctCoeffs));
        mfccs(1:actualCoeffs, i) = dctCoeffs(1:actualCoeffs);
        
        % Zero-pad if we need more coefficients than available
        if actualCoeffs < numCoeffs
            mfccs(actualCoeffs+1:numCoeffs, i) = 0;
        end
    end
end

%% MFCC impact analysis
function metrics = analyzeMFCCImpact(baselineMfccs, enhancedMfccs, speechMask, params)
    % Compute MFCC-based quality metrics
    
    % SNR improvement in MFCC domain
    baselineVar = var(baselineMfccs, [], 2);
    enhancedVar = var(enhancedMfccs, [], 2);
    snrImprovement = 10 * log10(mean(enhancedVar) / mean(baselineVar));
    
    % Spectral distortion
    mfccDiff = baselineMfccs - enhancedMfccs;
    spectralDistortion = sqrt(mean(mfccDiff(:).^2));
    
    % Dynamic range
    baselineRange = range(baselineMfccs, 2);
    enhancedRange = range(enhancedMfccs, 2);
    dynamicRangeChange = mean(enhancedRange) - mean(baselineRange);
    
    % Populate metrics structure
    metrics.snrImprovement = snrImprovement;
    metrics.spectralDistortion = spectralDistortion;
    metrics.dynamicRangeChange = dynamicRangeChange;
    metrics.baselineCoeffStats = struct('mean', mean(baselineMfccs, 2), 'std', std(baselineMfccs, [], 2));
    metrics.enhancedCoeffStats = struct('mean', mean(enhancedMfccs, 2), 'std', std(enhancedMfccs, [], 2));
end

%% SNR estimation
function snr = estimateSNR(signal, speechMask)
    speechSegments = signal(speechMask);
    noiseSegments = signal(~speechMask);
    
    if isempty(noiseSegments)
        snr = 20;  % Assume good SNR if no noise detected
        return;
    end
    
    speechPower = mean(speechSegments.^2);
    noisePower = mean(noiseSegments.^2);
    
    snr = 10 * log10(speechPower / noisePower);
end