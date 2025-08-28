% Simplified DAS Beamforming for Circular Array + Whisper ASR
% Optimized for 7-microphone circular array (6 outer + 1 center)
% Simple Delay-and-Sum beamformer for improved computational efficiency

clc; clear; close all;

%% Configuration
inputDir = '/media/niklas/SSD2/ind_beamforming/';
outputDir = '/media/niklas/SSD2/whisper_beamforming/';

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Array geometry - circular array with center microphone
radius = 0.035;           % 3.5 cm radius
nMics = 7;                % 6 outer + 1 center
channels = [1, 2, 3, 4, 5, 6, 7];  % All 7 channels
targetFs = 16000;         % Whisper sample rate
c = 343;                  % Speed of sound (m/s)

% DAS parameters
targetDirection = 0;      % Target direction in degrees (0 = front)
enableAdaptiveDAS = true; % Enable simple adaptive steering

%% Array geometry setup
arrayPos = getCircularArrayPositions(radius, nMics);

%% Process all segments
fprintf('Scanning for audio segments...\n');
segmentGroups = findAudioSegments(inputDir);
segmentIds = fieldnames(segmentGroups);

successCount = 0;
failCount = 0;

for segIdx = 1:length(segmentIds)
    segmentId = segmentIds{segIdx};
    files = segmentGroups.(segmentId);
    
    fprintf('\n=== Processing %d/%d: %s ===\n', segIdx, length(segmentIds), segmentId);
    
    try
        % Load multichannel audio
        [audioData, fs] = loadMultichannelAudio(files, channels, inputDir, targetFs);
        
        if isempty(audioData)
            fprintf('Failed to load audio for %s\n', segmentId);
            failCount = failCount + 1;
            continue;
        end
        
        % Estimate target direction if adaptive mode is enabled
        if enableAdaptiveDAS
            fprintf('  Estimating target direction...\n');
            targetDirection = estimateTargetDirection(audioData, arrayPos, fs, c);
            fprintf('  Target direction: %.1f degrees\n', targetDirection);
        end
        
        % Apply DAS beamforming
        fprintf('  Applying DAS beamforming...\n');
        beamformedSignal = applyDASBeamforming(audioData, arrayPos, targetDirection, fs, c);
        
        % Save output
        outputFilename = sprintf('%s_das.wav', segmentId);
        outputPath = fullfile(outputDir, outputFilename);
        audiowrite(outputPath, beamformedSignal, targetFs);
        
        successCount = successCount + 1;
        fprintf('Success: %s\n', outputFilename);
        
    catch ME
        fprintf('Error processing %s: %s\n', segmentId, ME.message);
        failCount = failCount + 1;
    end
end

fprintf('\n=== Complete ===\n');
fprintf('Success: %d, Failed: %d\n', successCount, failCount);

%% Functions

function arrayPos = getCircularArrayPositions(radius, nMics)
    % Create circular array geometry (6 outer + 1 center)
    % Mic 7 is center, Mics 1-6 are outer ring
    arrayPos = zeros(3, nMics);
    
    % Outer microphones (mics 1-6) equally spaced on circle
    for i = 1:6
        angle = (i-1) * 2*pi / 6;  % 6 outer mics, starting at 0°
        arrayPos(:, i) = [radius * cos(angle); radius * sin(angle); 0];
    end
    
    % Center microphone (mic 7) at origin
    arrayPos(:, 7) = [0; 0; 0];
end

function segmentGroups = findAudioSegments(inputDir)
    wavFiles = dir(fullfile(inputDir, '*.wav'));
    segmentGroups = struct();
    
    for i = 1:length(wavFiles)
        filename = wavFiles(i).name;
        tokens = regexp(filename, '(.+)\.CH(\d+)_(\d+)\.wav', 'tokens');
        
        if ~isempty(tokens)
            baseId = tokens{1}{1};
            segmentNum = tokens{1}{3};
            segmentId = [baseId '_' segmentNum];
            
            if ~isfield(segmentGroups, segmentId)
                segmentGroups.(segmentId) = {};
            end
            segmentGroups.(segmentId){end+1} = filename;
        end
    end
end

function [audioData, fs] = loadMultichannelAudio(files, channels, inputDir, targetFs)
    audioData = [];
    fs = targetFs;
    
    % Find files for each channel
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
        fprintf('Warning: Missing channels\n');
        return;
    end
    
    % Load and align all channels
    for i = 1:length(channels)
        [data, originalFs] = audioread(channelFiles{i});
        
        % Resample if needed
        if originalFs ~= targetFs
            data = resample(data, targetFs, originalFs);
        end
        
        if i == 1
            nSamples = length(data);
            audioData = zeros(nSamples, length(channels));
        else
            % Handle length mismatches
            minLength = min(nSamples, length(data));
            audioData = audioData(1:minLength, :);
            data = data(1:minLength);
            nSamples = minLength;
        end
        
        audioData(:, i) = data;
    end
    
    % Skip very short segments
    if nSamples < 0.5 * targetFs
        audioData = [];
    end
end

function targetDirection = estimateTargetDirection(audioData, arrayPos, fs, c)
    % Simple energy-based direction estimation using cross-correlation
    % This is much simpler than SRP-PHAT or MUSIC
    
    nMics = size(arrayPos, 2);
    azimuthRange = 0:10:350;  % Coarse search grid
    maxPower = -inf;
    targetDirection = 0;
    
    % Use a short segment for direction estimation
    segmentLength = min(round(1.0 * fs), size(audioData, 1));
    audioSegment = audioData(1:segmentLength, :);
    
    for azimuth = azimuthRange
        % Compute delays for this direction
        delays = computeDelays(arrayPos, azimuth, c);
        
        % Apply delays and sum
        aligned = applyDelays(audioSegment, delays, fs);
        beamformed = mean(aligned, 2);
        
        % Compute output power
        power = mean(beamformed.^2);
        
        if power > maxPower
            maxPower = power;
            targetDirection = azimuth;
        end
    end
end

function beamformedSignal = applyDASBeamforming(audioData, arrayPos, targetDirection, fs, c)
    % Simple Delay-and-Sum beamforming
    
    fprintf('    Computing delays for direction %.1f degrees...\n', targetDirection);
    
    % Compute time delays for each microphone
    delays = computeDelays(arrayPos, targetDirection, c);
    
    fprintf('    Applying delays and summing...\n');
    
    % Apply delays to align signals
    alignedSignals = applyDelays(audioData, delays, fs);
    
    % Sum aligned signals (simple average)
    beamformedSignal = mean(alignedSignals, 2);
    
    % Apply basic post-processing
    beamformedSignal = postProcessSignal(beamformedSignal, fs);
end

function delays = computeDelays(arrayPos, azimuth, c)
    % Compute time delays for each microphone relative to center mic
    nMics = size(arrayPos, 2);
    
    % Target direction vector
    azimuthRad = azimuth * pi / 180;
    targetDir = [cos(azimuthRad); sin(azimuthRad); 0];
    
    % Reference position (center microphone)
    refPos = arrayPos(:, 7);  % Center mic
    
    delays = zeros(nMics, 1);
    for i = 1:nMics
        % Vector from reference to microphone i
        deltaPos = arrayPos(:, i) - refPos;
        
        % Time delay (negative because we want to delay to align)
        delays(i) = -dot(deltaPos, targetDir) / c;
    end
end

function alignedSignals = applyDelays(audioData, delays, fs)
    % Apply fractional delays to align signals
    [nSamples, nMics] = size(audioData);
    alignedSignals = zeros(nSamples, nMics);
    
    for i = 1:nMics
        delaySamples = delays(i) * fs;
        
        if abs(delaySamples) < 0.1
            % No significant delay
            alignedSignals(:, i) = audioData(:, i);
        else
            % Apply fractional delay using linear interpolation
            alignedSignals(:, i) = applyFractionalDelay(audioData(:, i), delaySamples);
        end
    end
end

function delayedSignal = applyFractionalDelay(signal, delaySamples)
    % Apply fractional delay using linear interpolation
    
    % Separate integer and fractional parts
    intDelay = floor(delaySamples);
    fracDelay = delaySamples - intDelay;
    
    % Pad signal for delays
    maxDelay = abs(intDelay) + 1;
    paddedSignal = [zeros(maxDelay, 1); signal; zeros(maxDelay, 1)];
    
    % Apply integer delay
    if intDelay >= 0
        % Positive delay (shift right)
        startIdx = maxDelay + 1 + intDelay;
    else
        % Negative delay (shift left)
        startIdx = maxDelay + 1 + intDelay;
    end
    
    endIdx = startIdx + length(signal) - 1;
    
    % Ensure indices are valid
    startIdx = max(1, min(startIdx, length(paddedSignal)));
    endIdx = max(1, min(endIdx, length(paddedSignal)));
    
    if endIdx <= startIdx
        delayedSignal = zeros(size(signal));
        return;
    end
    
    % Extract integer-delayed signal
    intDelayedSignal = paddedSignal(startIdx:endIdx);
    
    % Apply fractional delay using linear interpolation
    if abs(fracDelay) < 1e-6
        delayedSignal = intDelayedSignal;
    else
        % Linear interpolation between adjacent samples
        if fracDelay > 0
            % Interpolate forward
            if endIdx < length(paddedSignal)
                nextSamples = paddedSignal(startIdx+1:endIdx+1);
                delayedSignal = (1 - fracDelay) * intDelayedSignal + fracDelay * nextSamples;
            else
                delayedSignal = intDelayedSignal;
            end
        else
            % Interpolate backward
            if startIdx > 1
                prevSamples = paddedSignal(startIdx-1:endIdx-1);
                delayedSignal = (1 + fracDelay) * intDelayedSignal - fracDelay * prevSamples;
            else
                delayedSignal = intDelayedSignal;
            end
        end
    end
    
    % Ensure output length matches input
    if length(delayedSignal) ~= length(signal)
        if length(delayedSignal) > length(signal)
            delayedSignal = delayedSignal(1:length(signal));
        else
            delayedSignal = [delayedSignal; zeros(length(signal) - length(delayedSignal), 1)];
        end
    end
end

function signal = postProcessSignal(signal, fs)
    % Simple post-processing to clean up the beamformed signal
    
    % Remove DC offset
    signal = signal - mean(signal);
    
    % Apply fade-in/out to prevent clicks
    fadeLength = round(0.01 * fs); % 10ms fade
    
    if length(signal) > 2 * fadeLength
        % Fade-in
        fadeIn = linspace(0, 1, fadeLength)';
        signal(1:fadeLength) = signal(1:fadeLength) .* fadeIn;
        
        % Fade-out
        fadeOut = linspace(1, 0, fadeLength)';
        signal(end-fadeLength+1:end) = signal(end-fadeLength+1:end) .* fadeOut;
    end
    
    % Simple high-pass filter to remove low-frequency artifacts
    if exist('butter', 'file')
        [b, a] = butter(2, 80/(fs/2), 'high');
        signal = filtfilt(b, a, signal);
    end
    
    % Normalize to prevent clipping
    maxVal = max(abs(signal));
    if maxVal > 0
        % Conservative normalization
        signal = signal * (0.8 / maxVal);
    end
    
    % Soft limiting to prevent any remaining artifacts
    signal = tanh(signal * 0.9);
end