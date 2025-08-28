% Streamlined MVDR Beamforming for Circular Array + Whisper ASR
% Optimized for 7-microphone circular array (6 outer + 1 center)

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

% MVDR parameters
regularization = 1e-3;    % Diagonal loading factor

% DOA estimation parameters
doaMethod = 'SRP-PHAT';   % 'SRP-PHAT' or 'MUSIC'
azimuthRange = 0:5:355;   % Search range in degrees
freqRange = [300, 3000];  % Frequency range for DOA (Hz)

%% Array geometry setup
arrayPos = getCircularArrayPositions(radius, nMics);

%% Process all segments
fprintf('Scanning for audio segments...\n');
segmentGroups = findAudioSegments(inputDir);
segmentIds = fieldnames(segmentGroups);
fprintf('Found %d segments to process\n', length(segmentIds));

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
        
        % Estimate DOA
        fprintf('  Estimating DOA using %s...\n', doaMethod);
        estimatedDOA = estimateDOA(audioData, arrayPos, azimuthRange, freqRange, ...
                                 fs, c, doaMethod);
        fprintf('  Estimated DOA: %.1f degrees\n', estimatedDOA);
        
        % Apply MVDR beamforming
        lookDirection = [estimatedDOA; 0];
        beamformedSignal = applyMVDRBeamforming(audioData, arrayPos, lookDirection, ...
                                              fs, c, regularization);
        
        % Save output
        outputFilename = sprintf('%s_mvdr.wav', segmentId);
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
    % Modified: Mic 7 is center, Mics 1-6 are outer ring
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

function beamformedSignal = applyMVDRBeamforming(audioData, arrayPos, lookDirection, fs, c, reg)
    [nSamples, nMics] = size(audioData);
    
    fprintf('  Performing VAD...\n');
    % Simple energy-based VAD
    frameSize = round(0.025 * fs);  % 25ms
    hopSize = round(0.01 * fs);     % 10ms
    
    speechMask = simpleVAD(audioData(:,7), frameSize, hopSize);
    noiseIndices = find(~speechMask);
    speechIndices = find(speechMask);
    
    fprintf('  Speech: %.1f%%, Noise: %.1f%%\n', ...
            100*length(speechIndices)/nSamples, 100*length(noiseIndices)/nSamples);
    
    if length(noiseIndices) < frameSize
        fprintf('  Insufficient noise data, using identity covariance\n');
        Rnn = eye(nMics) * reg;
    else
        fprintf('  Estimating noise covariance matrix...\n');
        % Estimate noise covariance from VAD-detected noise segments
        noiseData = audioData(noiseIndices, :);
        Rnn = (noiseData' * noiseData) / length(noiseIndices) + reg * eye(nMics);
    end
    
    fprintf('  Computing steering vector...\n');
    % Compute steering vector for look direction
    freq = 1000;  % Use 1kHz for steering vector (typical speech frequency)
    steeringVector = computeSteeringVector(arrayPos, lookDirection, freq, c);
    
    fprintf('  Computing MVDR weights...\n');
    % MVDR beamformer weights: w = (Rnn^-1 * a) / (a^H * Rnn^-1 * a)
    try
        RnnInv = inv(Rnn);
        w = (RnnInv * steeringVector) / (steeringVector' * RnnInv * steeringVector);
    catch
        fprintf('  Matrix inversion failed, using regularized version\n');
        RnnInv = inv(Rnn + reg * eye(nMics));
        w = (RnnInv * steeringVector) / (steeringVector' * RnnInv * steeringVector);
    end
    
    fprintf('  Applying beamforming...\n');
    % Apply beamforming weights
    beamformedSignal = real(audioData * conj(w));
    
    % Normalize output
    maxVal = max(abs(beamformedSignal));
    if maxVal > 0
        beamformedSignal = beamformedSignal / maxVal * 0.9;
    end
    
    % Light high-pass filtering (remove DC)
    [b, a] = butter(2, 50/(fs/2), 'high');
    beamformedSignal = filtfilt(b, a, beamformedSignal);
end

function speechMask = simpleVAD(signal, frameSize, hopSize)
    nSamples = length(signal);
    nFrames = floor((nSamples - frameSize) / hopSize) + 1;
    
    energy = zeros(nFrames, 1);
    
    % Compute frame energy
    for i = 1:nFrames
        startIdx = (i-1) * hopSize + 1;
        endIdx = min(startIdx + frameSize - 1, nSamples);
        frame = signal(startIdx:endIdx);
        energy(i) = sum(frame.^2);
    end
    
    % Adaptive threshold (more conservative than original)
    energyThreshold = prctile(energy, 40);
    speechFrames = energy > energyThreshold;
    
    % Median filtering to smooth decisions
    speechFrames = medfilt1(double(speechFrames), 3) > 0.5;
    
    % Convert frame decisions to sample mask
    speechMask = false(nSamples, 1);
    for i = 1:nFrames
        startIdx = (i-1) * hopSize + 1;
        endIdx = min(startIdx + frameSize - 1, nSamples);
        if speechFrames(i)
            speechMask(startIdx:endIdx) = true;
        end
    end
    
    % Ensure minimum speech detection
    if sum(speechMask) < 0.1 * nSamples
        energyThreshold = prctile(energy, 25);
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

function estimatedDOA = estimateDOA(audioData, arrayPos, azimuthRange, freqRange, fs, c, method)
    [nSamples, nMics] = size(audioData);
    
    % Use speech segments for DOA estimation
    frameSize = round(0.025 * fs);
    hopSize = round(0.01 * fs);
    speechMask = simpleVAD(audioData(:,1), frameSize, hopSize);
    speechIndices = find(speechMask);
    
    if length(speechIndices) < frameSize
        fprintf('    Insufficient speech data, using broadside direction\n');
        estimatedDOA = 0;
        return;
    end
    
    % Use a representative speech segment (up to 2 seconds)
    maxSamples = min(2 * fs, length(speechIndices));
    sampleIndices = speechIndices(1:min(end, maxSamples));
    speechData = audioData(sampleIndices, :);
    
    switch upper(method)
        case 'SRP-PHAT'
            estimatedDOA = srpPhatDOA(speechData, arrayPos, azimuthRange, freqRange, fs, c);
        case 'MUSIC'
            estimatedDOA = musicDOA(speechData, arrayPos, azimuthRange, freqRange, fs, c);
        otherwise
            error('Unknown DOA method: %s', method);
    end
end

function estimatedDOA = srpPhatDOA(audioData, arrayPos, azimuthRange, freqRange, fs, c)
    % SRP-PHAT DOA estimation optimized for circular arrays
    nMics = size(arrayPos, 2);
    nFFT = 512;
    srpPower = zeros(size(azimuthRange));
    
    % Frequency bins of interest
    freqBins = round(freqRange * nFFT / fs) + 1;
    freqBins = freqBins(1):freqBins(2);
    freqBins = freqBins(freqBins <= nFFT/2 + 1);
    
    for azIdx = 1:length(azimuthRange)
        azimuth = azimuthRange(azIdx);
        power = 0;
        
        % Compute steering delays for this direction
        lookDir = [cosd(azimuth); sind(azimuth); 0];
        refPos = arrayPos(:, 7);
        delays = zeros(nMics, 1);
        
        for m = 1:nMics
            deltaPos = arrayPos(:, m) - refPos;
            delays(m) = dot(deltaPos, lookDir) / c;
        end
        
        % Process in frames
        frameSize = round(0.032 * fs);  % 32ms frames
        hopSize = round(0.016 * fs);    % 16ms hop
        nFrames = floor((size(audioData, 1) - frameSize) / hopSize) + 1;
        
        for frame = 1:min(nFrames, 50)  % Limit frames for speed
            startIdx = (frame-1) * hopSize + 1;
            endIdx = startIdx + frameSize - 1;
            
            if endIdx > size(audioData, 1)
                break;
            end
            
            frameData = audioData(startIdx:endIdx, :);
            
            % FFT of each channel
            X = fft(frameData, nFFT, 1);
            
            % SRP-PHAT computation
            framePower = 0;
            pairCount = 0;
            
            for m1 = 1:nMics-1
                for m2 = m1+1:nMics
                    % Cross-power spectrum with PHAT weighting
                    X1 = X(:, m1);
                    X2 = X(:, m2);
                    
                    G12 = X1 .* conj(X2);
                    W = abs(G12);  % PHAT weighting
                    W(W < eps) = eps;  % Avoid division by zero
                    G12_weighted = G12 ./ W;
                    
                    % Apply steering delay
                    deltaDelay = delays(m2) - delays(m1);
                    
                    for k = freqBins
                        freq = (k-1) * fs / nFFT;
                        phaseShift = exp(-1j * 2 * pi * freq * deltaDelay);
                        framePower = framePower + real(G12_weighted(k) * phaseShift);
                    end
                    
                    pairCount = pairCount + 1;
                end
            end
            
            power = power + framePower / pairCount;
        end
        
        srpPower(azIdx) = power;
    end
    
    % Find peak
    [~, maxIdx] = max(srpPower);
    estimatedDOA = azimuthRange(maxIdx);
end

function estimatedDOA = musicDOA(audioData, arrayPos, azimuthRange, freqRange, fs, c)
    % MUSIC DOA estimation
    nMics = size(arrayPos, 2);
    nSources = 1;  % Assume single source
    
    % Compute sample covariance matrix
    R = (audioData' * audioData) / size(audioData, 1);
    
    % Add diagonal loading for robustness
    R = R + 1e-3 * eye(nMics);
    
    % Eigendecomposition
    [V, D] = eig(R);
    [eigenvals, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % Noise subspace (assuming nSources sources)
    noiseSubspace = V(:, nSources+1:end);
    
    % MUSIC spectrum
    musicSpectrum = zeros(size(azimuthRange));
    
    for azIdx = 1:length(azimuthRange)
        azimuth = azimuthRange(azIdx);
        
        % Compute steering vector at center frequency
        centerFreq = mean(freqRange);
        steeringVec = computeSteeringVector(arrayPos, [azimuth; 0], centerFreq, c);
        
        % MUSIC pseudo-spectrum
        denominator = steeringVec' * (noiseSubspace * noiseSubspace') * steeringVec;
        musicSpectrum(azIdx) = 1 / real(denominator);
    end
    
    % Find peak
    [~, maxIdx] = max(musicSpectrum);
    estimatedDOA = azimuthRange(maxIdx);
end

function steeringVector = computeSteeringVector(arrayPos, lookDirection, freq, c)
    nMics = size(arrayPos, 2);
    
    % Convert look direction to unit vector
    azimuth = lookDirection(1) * pi/180;  % Convert to radians
    elevation = lookDirection(2) * pi/180;
    
    lookDir = [cos(elevation) * cos(azimuth);
               cos(elevation) * sin(azimuth);
               sin(elevation)];
    
    % Compute time delays relative to first microphone
    refPos = arrayPos(:, 1);  % Use first mic as reference
    steeringVector = zeros(nMics, 1);
    
    for i = 1:nMics
        % Time delay relative to reference microphone
        deltaPos = arrayPos(:, i) - refPos;
        timeDelay = dot(deltaPos, lookDir) / c;
        
        % Phase shift (complex exponential)
        phaseShift = -2 * pi * freq * timeDelay;
        steeringVector(i) = exp(1j * phaseShift);
    end
end