% Streamlined MVDR Beamforming for Circular Array + Whisper ASR
% Optimized for 7-microphone circular array (6 outer + 1 center)
% Enhanced with frequency-domain MVDR beamforming using Signal Processing Toolbox

clc; clear; close all;

%% Configuration
inputDir = '/media/niklas/SSD2/ind_beamforming/';
outputDir = '/media/niklas/SSD2/whisper_beamforming360/';

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
regularization = 5e-3;    % Diagonal loading factor
useFrequencyDomain = true; % Set to false to use original single-frequency method

% DOA estimation parameters
doaMethod = 'SRP-PHAT';   % 'SRP-PHAT' or 'MUSIC'
azimuthRange = 0:2:358;   % Search range in degrees
freqRange = [200, 4000];  % Frequency range for DOA (Hz)

%% Array geometry setup
arrayPos = getCircularArrayPositions(radius, nMics);

%% Process all segments - Generate 360-degree beamformed audio
fprintf('Scanning for audio segments...\n');
segmentGroups = findAudioSegments(inputDir);
segmentIds = fieldnames(segmentGroups);

% Define 360-degree test angles
testAngles = 0:1:359;  % Every degree from 0 to 359

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
        
        % Create output subdirectory for this segment
        segmentOutputDir = fullfile(outputDir, segmentId);
        if ~exist(segmentOutputDir, 'dir')
            mkdir(segmentOutputDir);
        end
        
        % Generate beamformed audio for each test angle (NO DOA ESTIMATION)
        fprintf('  Generating 360-degree beamformed audio...\n');
        
        for angleIdx = 1:length(testAngles)
            testAngle = testAngles(angleIdx);
            lookDirection = [testAngle; 0];  % Force beamformer to this angle

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

function beamformedSignal = applyFrequencyDomainMVDR(audioData, arrayPos, lookDirection, fs, c, reg)
    % Frequency-domain MVDR beamforming using Signal Processing Toolbox
    % Uses STFT with proper windowing and overlap-add reconstruction
    
    [nSamples, nMics] = size(audioData);
    
    fprintf('    Performing VAD...\n');
    % Use improved VAD with spectraC features
    speechMask = improvedVAD(audioData(:,7), fs);
    noiseIndices = find(~speechMask);
    speechIndices = find(speechMask);
    
    fprintf('    Speech: %.1f%%, Noise: %.1f%%\n', ...
            100*length(speechIndices)/nSamples, 100*length(noiseIndices)/nSamples);
    
    % STFT parameters optimized for speech
     % 32ms window
    windowLength = 2^nextpow2(round(0.025 * fs));  %
    overlap = round(0.75 * windowLength);  % 75% overlap
    nFFT = 2^nextpow2(windowLength);  % Next power of 2
    
    % Use Signal Processing Toolbox STFT
    fprintf('    Computing STFT for all channels...\n');
    [S, F, T] = stft(audioData, fs, 'Window', hann(windowLength, 'periodic'), ...
                     'OverlapLength', overlap, 'FFTLength', nFFT);
    
    [nFreqs, nFrames, nMics] = size(S);
    
    % Estimate noise covariance matrices using improved method
    fprintf('    Estimating frequency-dependent noise covariance...\n');
    Rnn = estimateNoiseCovariance(S, speechMask, fs, T, reg);
    
    % Compute frequency-dependent steering vectors and MVDR weights
    fprintf('    Computing frequency-dependent MVDR weights...\n');
    W = complex(zeros(nMics, nFreqs));
    
    for k = 1:nFreqs
        freq = F(k);
        
        % Skip DC and very low frequencies
        if freq < 50
            W(:, k) = [1; zeros(nMics-1, 1)] / nMics;  % Simple average
            continue;
        end
        
        % Compute steering vector for this frequency
        steeringVector = computeSteeringVector(arrayPos, lookDirection, freq, c);
        
        % MVDR weights: w = (Rnn^-1 * a) / (a^H * Rnn^-1 * a)
        try
            RnnInv = inv(squeeze(Rnn(:,:,k)));
            denominator = steeringVector' * RnnInv * steeringVector;
            if abs(denominator) > eps
                W(:, k) = (RnnInv * steeringVector) / denominator;
            else
                W(:, k) = steeringVector / (steeringVector' * steeringVector);
            end
        catch
            % Fallback to delay-and-sum if inversion fails
            W(:, k) = steeringVector / (steeringVector' * steeringVector);
        end
    end
    
    % Apply frequency-domain beamforming
    fprintf('    Applying frequency-domain beamforming...\n');
    beamformSTFT = complex(zeros(nFreqs, nFrames));
    
    for frameIdx = 1:nFrames
        for k = 1:nFreqs
            x_k = squeeze(S(k, frameIdx, :));  % Frequency bin across all mics
            beamformedSTFT(k, frameIdx) = W(:, k)' * x_k;
        end
    end
    
    % Convert back to time domain using inverse STFT
    fprintf('    Converting back to time domain...\n');
    % Make sure ISTFT uses exactly the same parameters as STFT
       % Convert back to time domain using inverse STFT
    fprintf('    Converting back to time domain...\n');
    beamformedSignal = istft(beamformedSTFT, fs, ...
    'Window', hann(windowLength, 'periodic'), ...
    'OverlapLength', overlap, ...
    'FFTLength', nFFT, ...
    'ConjugateSymmetric', true);
    %beamformedSignal = istft(beamformedSTFT, fs, 'Window', hann(windowLength, 'periodic'), ...'OverlapLength', overlap, 'FFTLength', nFFT, ...'ConjugateSymmetric', true);
    beamformedSignal = beamformedSignal(:);
    % Ensure output length matches input
    originalLength = size(audioData, 1);

% After ISTFT, use proper windowing for length adjustment
    originalLength = size(audioData, 1);
    if length(beamformedSignal) > originalLength
    % Apply longer fade-out to avoid clicks
        fadeLength = min(round(0.01 * fs), (length(beamformedSignal) - originalLength) + round(0.005 * fs));
        if fadeLength > 0
            endIdx = min(originalLength + fadeLength - 1, length(beamformedSignal));
            fadeWindow = linspace(1, 0, endIdx - originalLength + 1)';
            beamformedSignal(originalLength:endIdx) = beamformedSignal(originalLength:endIdx) .* fadeWindow;
        end
        beamformedSignal = beamformedSignal(1:originalLength);
    elseif length(beamformedSignal) < originalLength
    % Pad with small fade-in instead of zeros
        padLength = originalLength - length(beamformedSignal);
        fadePad = linspace(0, 0.01, min(padLength, round(0.005*fs)))';
        padding = [fadePad; zeros(padLength - length(fadePad), 1)];
        beamformedSignal = [beamformedSignal; padding];
    end
  
    % Apply post-processing
    beamformedSignal = postProcessSignal(beamformedSignal, fs);
end

function Rnn = estimateNoiseCovariance(S, speechMask, fs, T, reg)
    [nFreqs, nFrames, nMics] = size(S);
    
    % Convert speech mask with better temporal alignment
    frameRate = fs / (round(0.75 * 2^nextpow2(round(0.032 * fs))) - round(0.25 * 2^nextpow2(round(0.032 * fs))));
    speechFrames = false(nFrames, 1);
    
    for i = 1:nFrames
        timeStart = max(1, round((i-1) * frameRate * (T(2) - T(1))) + 1);
        timeEnd = min(length(speechMask), timeStart + round(frameRate * (T(2) - T(1))));
        % Use majority vote for frame classification
        speechFrames(i) = mean(speechMask(timeStart:timeEnd)) > 0.3;
    end
    
    noiseFrames = ~speechFrames;
    
    % Apply temporal smoothing to avoid isolated noise frames
    noiseFrames = medfilt1(double(noiseFrames), 5) > 0.5;
    
    Rnn = complex(zeros(nMics, nMics, nFreqs));
    
    if sum(noiseFrames) < max(10, 0.1 * nFrames)  % Need at least 10% noise frames
        fprintf('    Using diagonal loading only (insufficient noise)\n');
        for k = 1:nFreqs
            Rnn(:,:,k) = eye(nMics) * (reg * 10);  % Increase regularization
        end
        return;
    end
    
    % Frequency-dependent regularization
    for k = 1:nFreqs
        noiseData = squeeze(S(k, noiseFrames, :));
        
        if size(noiseData, 1) > 2
            R_sample = (noiseData' * noiseData) / size(noiseData, 1);
            % Frequency-dependent regularization
            freq = (k-1) * fs / (2 * (nFreqs-1));
            freqReg = reg * (1 + 0.5 * exp(-freq/1000));  % More reg at low freq
            Rnn(:,:,k) = R_sample + freqReg * eye(nMics);
        else
            Rnn(:,:,k) = eye(nMics) * reg * 5;
        end
    end
end
function speechMask = improvedVAD(signal, fs)
    % Improved VAD using spectral features and Signal Processing Toolbox
    
    % STFT for spectral analysis
    windowLength = round(0.032 * fs);  % 25ms
    overlap = round(0.75 * windowLength);        % 10ms
    nFFT = 4 * windowLength; 
    [S, F, T] = stft(signal, fs, 'Window', hann(windowLength, 'periodic'), ...
                     'OverlapLength', overlap);
    
    % Compute spectral features
    spectralCentroid = computeSpectralCentroid(S, F);
    spectralEntropy = computeSpectralEntropy(abs(S));
    energy = sum(abs(S).^2, 1);
    
    % Normalize features
    spectralCentroid = (spectralCentroid - mean(spectralCentroid)) / std(spectralCentroid);
    spectralEntropy = (spectralEntropy - mean(spectralEntropy)) / std(spectralEntropy);
    energy = (energy - mean(energy)) / std(energy);
    
    % Combined feature score
    vadScore = 0.6 * energy + 0.2 * spectralCentroid + 0.2 * spectralEntropy;
    
    % Adaptive threshold
    threshold = prctile(vadScore, 25);
    speechFrames = vadScore > threshold;
    
    % Smooth decisions with median filter
    speechFrames = medfilt1(double(speechFrames), 3) > 0.5;
    
    % Convert frame decisions to sample mask
    speechMask = false(length(signal), 1);
    for i = 1:length(speechFrames)
        startIdx = round((i-1) * (T(2) - T(1)) * fs) + 1;
        endIdx = min(startIdx + windowLength - 1, length(signal));
        if speechFrames(i) && startIdx <= length(signal)
            speechMask(startIdx:endIdx) = true;
        end
    end
end

function centroid = computeSpectralCentroid(S, F)
    % Compute spectral centroid for each frame
    magnitude = abs(S);
    totalEnergy = sum(magnitude, 1);
    
    % Avoid division by zero
    totalEnergy(totalEnergy < eps) = eps;
    
    % Weighted frequency sum
    weightedSum = sum(magnitude .* repmat(F, 1, size(S, 2)), 1);
    centroid = weightedSum ./ totalEnergy;
end

function entropy = computeSpectralEntropy(magnitude)
    % Compute spectral entropy for each frame
    % Normalize magnitude to probability
    totalEnergy = sum(magnitude, 1);
    totalEnergy(totalEnergy < eps) = eps;
    
    prob = magnitude ./ repmat(totalEnergy, size(magnitude, 1), 1);
    prob(prob < eps) = eps;  % Avoid log(0)
    
    % Compute entropy
    entropy = -sum(prob .* log(prob), 1);
end

function beamformedSignal = applyMVDRBeamforming(audioData, arrayPos, lookDirection, fs, c, reg)
    % Original single-frequency MVDR beamforming with Signal Processing Toolbox improvements
    [nSamples, nMics] = size(audioData);
    
    fprintf('    Performing improved VAD...\n');
    speechMask = improvedVAD(audioData(:,7), fs);
    noiseIndices = find(~speechMask);
    speechIndices = find(speechMask);
    
    fprintf('    Speech: %.1f%%, Noise: %.1f%%\n', ...
            100*length(speechIndices)/nSamples, 100*length(noiseIndices)/nSamples);
    
    if length(noiseIndices) < round(0.025 * fs)
        fprintf('    Insufficient noise data, using identity covariance\n');
        Rnn = eye(nMics) * reg;
    else
        fprintf('    Estimating noise covariance matrix...\n');
        % Use robust covariance estimation
        noiseData = audioData(noiseIndices, :);
        Rnn = robustCov(noiseData) + reg * eye(nMics);
    end
    
    fprintf('    Computing steering vector...\n');
    % Compute steering vector for look direction
    freq = 1000;  % Use 1kHz for steering vector
    steeringVector = computeSteeringVector(arrayPos, lookDirection, freq, c);
    
    fprintf('    Computing MVDR weights...\n');
    % MVDR beamformer weights with improved numerical stability
    try
        [U, S_eig, V] = svd(Rnn);
        % Regularized pseudo-inverse
        S_inv = diag(1 ./ (diag(S_eig) + reg));
        RnnInv = V * S_inv * U';
        
        w = (RnnInv * steeringVector) / (steeringVector' * RnnInv * steeringVector);
    catch
        fprintf('    Using fallback computation\n');
        w = steeringVector / (steeringVector' * steeringVector);
    end
    
    fprintf('    Applying beamforming...\n');
    % Apply beamforming weights
    beamformedSignal = real(audioData * conj(w));
    
    % Apply post-processing
    beamformedSignal = postProcessSignal(beamformedSignal, fs);
end

function R = robustCov(data)
    % Robust covariance estimation using Median Absolute Deviation
    [n, p] = size(data);
    
    % Center the data using median
    medianData = median(data, 1);
    centeredData = data - repmat(medianData, n, 1);
    
    % Compute MAD-based scaling
    mad_vals = mad(centeredData, 1, 1);  % Median absolute deviation
    mad_vals(mad_vals < eps) = 1;  % Avoid division by zero
    
    scaledData = centeredData ./ repmat(mad_vals, n, 1);
    
    % Compute robust covariance
    R = (scaledData' * scaledData) / (n - 1);
    
    % Scale back
    R = R .* (mad_vals' * mad_vals);
end

function signal = postProcessSignal(signal, fs)
    % Enhanced post-processing to prevent artifacts and improve quality
    
    % 1. Remove any DC offset first
    signal = signal - mean(signal);
    
    % 2. Longer fade-in/out with smooth transitions
    fadeLength = round(0.02 * fs); % 20ms fade
    
    if length(signal) > 2 * fadeLength
        % Smooth fade-in with raised cosine
        fadeIn = 0.5 * (1 - cos(linspace(0, pi, fadeLength)))';
        signal(1:fadeLength) = signal(1:fadeLength) .* fadeIn;
        
        % Smooth fade-out with raised cosine
        fadeOut = 0.5 * (1 + cos(linspace(0, pi, fadeLength)))';
        signal(end-fadeLength+1:end) = signal(end-fadeLength+1:end) .* fadeOut;
    end
    
    % 3. Multi-stage filtering
    % High-pass filter to remove low-frequency artifacts
    [b_hp, a_hp] = butter(4, 100/(fs/2), 'high');
    signal = filtfilt(b_hp, a_hp, signal);
    
    % Gentle low-pass to reduce high-frequency artifacts
    [b_lp, a_lp] = butter(4, 7000/(fs/2), 'low');
    signal = filtfilt(b_lp, a_lp, signal);
    
    % 4. Spectral subtraction for residual noise
    signal = spectralSubtraction(signal, fs);
    
    % 5. Conservative normalization
    maxAbs = max(abs(signal));
    if maxAbs > 0
        targetRMS = 0.02; % Even lower target
        currentRMS = sqrt(mean(signal.^2));
        if currentRMS > 0
            gainFactor = min(targetRMS / currentRMS, 0.7 / maxAbs);
            signal = signal * gainFactor;
        end
    end
    
    % 6. Final soft clipping with smooth transition
    signal = tanh(signal * 0.9) * 0.8;
end

function cleanSignal = spectralSubtraction(signal, fs)
    % Simple spectral subtraction for residual noise reduction
    windowLength = round(0.025 * fs);
    overlap = round(0.5 * windowLength);
    
    [S, F, T] = stft(signal, fs, 'Window', hann(windowLength), 'OverlapLength', overlap);
    
    % Estimate noise spectrum from first and last 10% of frames
    noiseFrames = [1:round(0.1*size(S,2)), round(0.9*size(S,2)):size(S,2)];
    noiseSpectrum = mean(abs(S(:, noiseFrames)).^2, 2);
    
    % Spectral subtraction with over-subtraction factor
    alpha = 2.0; % Over-subtraction factor
    beta = 0.01; % Spectral floor
    
    magnitude = abs(S);
    phase = angle(S);
    
    % Apply spectral subtraction
    enhancedMagnitude = magnitude.^2 - alpha * repmat(noiseSpectrum, 1, size(S,2));
    enhancedMagnitude = max(enhancedMagnitude, beta * magnitude.^2);
    enhancedMagnitude = sqrt(enhancedMagnitude);
    
    % Reconstruct signal
    enhancedS = enhancedMagnitude .* exp(1j * phase);
    cleanSignal = istft(enhancedS, fs, 'Window', hann(windowLength), 'OverlapLength', overlap);
    
    % Ensure same length
    if length(cleanSignal) ~= length(signal)
        cleanSignal = cleanSignal(1:min(length(cleanSignal), length(signal)));
        if length(cleanSignal) < length(signal)
            cleanSignal = [cleanSignal; zeros(length(signal) - length(cleanSignal), 1)];
        end
    end
end
function estimatedDOA = estimateDOA(audioData, arrayPos, azimuthRange, freqRange, fs, c, method)
    [nSamples, nMics] = size(audioData);
    
    % Use improved speech detection
    speechMask = improvedVAD(audioData(:,1), fs);
    speechIndices = find(speechMask);
    
    if length(speechIndices) < round(0.1 * fs)
        fprintf('      Insufficient speech data, using broadside direction\n');
        estimatedDOA = 0;
        return;
    end
    
    % Use a representative speech segment
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
    % SRP-PHAT using Signal Processing Toolbox STFT
    nMics = size(arrayPos, 2);
    srpPower = zeros(size(azimuthRange));
    
    % Use STFT for better frequency resolution
    windowLength = round(0.025 * fs);
    overlap = round(0.01 * windowLength);
    
    [S, F, T] = stft(audioData, fs, 'Window', hann(windowLength, 'periodic'), ...
                     'OverlapLength', overlap);
    
    % Frequency bins of interest
    freqBins = (F >= freqRange(1)) & (F <= freqRange(2));
    
    for azIdx = 1:length(azimuthRange)
        azimuth = azimuthRange(azIdx);
        
        % Compute steering delays
        lookDir = [cosd(azimuth); sind(azimuth); 0];
        refPos = arrayPos(:, 7);  % Use center mic as reference
        delays = zeros(nMics, 1);
        
        for m = 1:nMics
            deltaPos = arrayPos(:, m) - refPos;
            delays(m) = dot(deltaPos, lookDir) / c;
        end
        
        % SRP-PHAT computation using cross-correlation
        power = 0;
        pairCount = 0;
        
        % Limit time frames for computational efficiency
        maxFrames = min(size(S, 2), 100);
        
        for m1 = 1:nMics-1
            for m2 = m1+1:nMics
                % Cross-power spectrum with PHAT weighting
                X1 = squeeze(S(:, 1:maxFrames, m1));
                X2 = squeeze(S(:, 1:maxFrames, m2));
                
                G12 = X1 .* conj(X2);
                W = abs(G12);
                W(W < eps) = eps;
                G12_weighted = G12 ./ W;
                
                % Apply steering delays
                deltaDelay = delays(m2) - delays(m1);
                
                for k = find(freqBins)'
                    freq = F(k);
                    phaseShift = exp(-1j * 2 * pi * freq * deltaDelay);
                    power = power + sum(real(G12_weighted(k, :) * phaseShift));
                end
                
                pairCount = pairCount + 1;
            end
        end
        
        srpPower(azIdx) = power / pairCount;
    end
    
    % Find peak with smoothing
    srpPower = smoothdata(srpPower, 'gaussian', 3);
    [~, maxIdx] = max(srpPower);
    estimatedDOA = azimuthRange(maxIdx);
end

function estimatedDOA = musicDOA(audioData, arrayPos, azimuthRange, freqRange, fs, c)
    % MUSIC DOA estimation with improved eigenvalue analysis
    nMics = size(arrayPos, 2);
    
    % Use only speech segments for covariance estimation
    speechMask = improvedVAD(audioData(:,1), fs);
    speechIndices = find(speechMask);
    
    if length(speechIndices) > round(0.5 * fs)
        speechData = audioData(speechIndices(1:round(0.5 * fs)), :);
    else
        speechData = audioData(speechIndices, :);
    end
    
    % Compute sample covariance matrix
    R = robustCov(speechData);
    
    % Eigendecomposition
    [V, D] = eig(R);
    [eigenvals, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % Estimate number of sources using AIC/MDL or assume 1
    nSources = 1;
    
    % Enhanced source detection (optional)
    if nMics > 3
        % Simple eigenvalue-based source detection
        eigenRatio = eigenvals(1:end-1) ./ eigenvals(2:end);
        [~, maxRatioIdx] = max(eigenRatio);
        nSources = min(maxRatioIdx, 2);  % Limit to 2 sources max
    end
    
    % Noise subspace
    noiseSubspace = V(:, nSources+1:end);
    
    % MUSIC spectrum
    musicSpectrum = zeros(size(azimuthRange));
    
    for azIdx = 1:length(azimuthRange)
        azimuth = azimuthRange(azIdx);
        
        % Use multiple frequencies for robustness
        spectrum_sum = 0;
        freq_count = 0;
        
        for freq = freqRange(1):100:freqRange(2)
            steeringVec = computeSteeringVector(arrayPos, [azimuth; 0], freq, c);
            denominator = steeringVec' * (noiseSubspace * noiseSubspace') * steeringVec;
            spectrum_sum = spectrum_sum + 1 / max(real(denominator), eps);
            freq_count = freq_count + 1;
        end
        
        musicSpectrum(azIdx) = spectrum_sum / freq_count;
    end
    
    % Find peak with smoothing
    musicSpectrum = smoothdata(musicSpectrum, 'gaussian', 3);
    [~, maxIdx] = max(musicSpectrum);
    estimatedDOA = azimuthRange(maxIdx);
end

function steeringVector = computeSteeringVector(arrayPos, lookDirection, freq, c)
    nMics = size(arrayPos, 2);
    
    % Convert look direction to unit vector
    azimuth = lookDirection(1) * pi/180;
    elevation = lookDirection(2) * pi/180;
    
    lookDir = [cos(elevation) * cos(azimuth);
               cos(elevation) * sin(azimuth);
               sin(elevation)];
    
    % Use center microphone as reference for circular array
    refPos = arrayPos(:, 7);  % Center microphone
    steeringVector = zeros(nMics, 1);
    
    for i = 1:nMics
        % Time delay relative to reference microphone
        deltaPos = arrayPos(:, i) - refPos;
        timeDelay = dot(deltaPos, lookDir) / c;
        
        % Phase shift
        phaseShift = -2 * pi * freq * timeDelay;
        steeringVector(i) = exp(1j * phaseShift);
    end
end
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
        
        % Create output subdirectory for this segment
        segmentOutputDir = fullfile(outputDir, segmentId);
        if ~exist(segmentOutputDir, 'dir')
            mkdir(segmentOutputDir);
        end
        
        % Generate beamformed audio for each test angle (NO DOA ESTIMATION)
        fprintf('  Generating 360-degree beamformed audio...\n');
        
        for angleIdx = 1:length(testAngles)
            testAngle = testAngles(angleIdx);
            lookDirection = [testAngle; 0];  % Force beamformer to this angle
            
            % Apply MVDR beamforming at this specific angle
            if useFrequencyDomain
                beamformedSignal = applyFrequencyDomainMVDR(audioData, arrayPos, lookDirection, ...
                                                          fs, c, regularization);
            else
                beamformedSignal = applyMVDRBeamforming(audioData, arrayPos, lookDirection, ...
                                                      fs, c, regularization);
            end
            
            % Save with angle-specific filename
            outputFilename = sprintf('%s_mvdr_%03d.wav', segmentId, testAngle);
            outputPath = fullfile(segmentOutputDir, outputFilename);
            audiowrite(outputPath, beamformedSignal, targetFs);
            
            % Progress indicator every 30 degrees
            if mod(testAngle, 30) == 0
                fprintf('    Completed angle: %d degrees\n', testAngle);
            end
        end
        
        successCount = successCount + 1;
        fprintf('Success: Generated %d files for %s\n', length(testAngles), segmentId);
        
    catch ME
        fprintf('Error processing %s: %s\n', segmentId, ME.message);
        failCount = failCount + 1;
    end
end