% Streamlined MVDR Beamforming for Circular Array + Whisper ASR
% Optimized for 7-microphone circular array (6 outer + 1 center)
% Enhanced with frequency-domain MVDR beamforming using Signal Processing Toolbox
set(groot, 'DefaultFigureRenderer', 'painters');  % Try different renderer
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
regularization = 5e-3;    % Diagonal loading factor
useFrequencyDomain = true; % Set to false to use original single-frequency method

% DOA estimation parameters
doaMethod = 'SRP-PHAT';   % 'SRP-PHAT' or 'MUSIC'
azimuthRange = 0:2:358;   % Search range in degrees
freqRange = [200, 4000];  % Frequency range for DOA (Hz)

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
        
        % Estimate DOA
        fprintf('  Estimating DOA using %s...\n', doaMethod);
        estimatedDOA = estimateDOA(audioData, arrayPos, azimuthRange, freqRange, ...
                                 fs, c, doaMethod);
        fprintf('  Estimated DOA: %.1f degrees\n', estimatedDOA);
        
        % Apply MVDR beamforming
        lookDirection = [estimatedDOA; 0];
        
        if useFrequencyDomain
            fprintf('  Using frequency-domain MVDR beamforming...\n');
            beamformedSignal = applyFrequencyDomainMVDR(audioData, arrayPos, lookDirection, ...
                                                      fs, c, regularization);
        else
            fprintf('  Using single-frequency MVDR beamforming...\n');
            beamformedSignal = applyMVDRBeamforming(audioData, arrayPos, lookDirection, ...
                                                  fs, c, regularization);
        end
        
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

function beamformedSignal = applyFrequencyDomainMVDR(audioData, arrayPos, lookDirection, fs, c, reg)
    % Frequency-domain MVDR beamforming using Signal Processing Toolbox
    % Uses STFT with proper windowing and overlap-add reconstruction
    
    [nSamples, nMics] = size(audioData);
    
    fprintf('    Performing VAD...\n');
    % Use improved VAD with spectral features
    speechMask = improvedVAD(audioData(:,7), fs);
    noiseIndices = find(~speechMask);
    speechIndices = find(speechMask);
    
    fprintf('    Speech: %.1f%%, Noise: %.1f%%\n', ...
            100*length(speechIndices)/nSamples, 100*length(noiseIndices)/nSamples);
    
    % STFT parameters optimized for speech
    windowLength = 2^nextpow2(round(0.025 * fs));  % 32ms window
    overlap = round(0.75 * windowLength);  % 75% overlap
    nFFT = 2^nextpow2(windowLength);  % Next power of 2
    padLength = windowLength;
    paddedAudio = [zeros(padLength, nMics); audioData; zeros(padLength, nMics)];
    % Use Signal Processing Toolbox STFT
    fprintf('    Computing STFT for all channels...\n');
    [S, F, T] = stft(paddedAudio, fs, 'Window', hann(windowLength, 'periodic'), ...
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
    beamformedSTFT = complex(zeros(nFreqs, nFrames));
    
    for frameIdx = 1:nFrames
        for k = 1:nFreqs
            x_k = squeeze(S(k, frameIdx, :));  % Frequency bin across all mics
            beamformedSTFT(k, frameIdx) = W(:, k)' * x_k;
        end
    end
    
    % Convert back to time domain using inverse STFT
    fprintf('    Converting back to time domain...\n');
    
    % Ensure the STFT matrix has the correct conjugate symmetry for real output
    % This is crucial for ISTFT to work properly
    if mod(nFFT, 2) == 0
        % Even FFT length - ensure Nyquist bin is real
        beamformedSTFT(end, :) = real(beamformedSTFT(end, :));
    end
    
    % Ensure DC component is real
    beamformedSTFT(1, :) = real(beamformedSTFT(1, :));
    
    try
        % Use ISTFT with exactly matching parameters
        beamformedSignal = istft(beamformedSTFT, fs, ...
                                'Window', hann(windowLength, 'periodic'), ...
                                'OverlapLength', overlap, ...
                                'FFTLength', nFFT, ...
                                'ConjugateSymmetric', true);
    catch ME
        fprintf('    ISTFT failed (%s), trying alternative approach...\n', ME.message);
        
        % Alternative: Manual overlap-add reconstruction
        beamformedSignal = manualIFFTReconstruction(beamformedSTFT, windowLength, overlap, nFFT);
    end
    
    % Ensure output is real and column vector
    beamformedSignal = real(beamformedSignal(:));
    
    % Ensure output length matches input
    originalLength = size(audioData, 1);
    
    % After ISTFT, use proper windowing for length adjustmen  
% Instead of complex fade logic, do simple truncation for ASR
    if length(beamformedSignal) > 2 * padLength
    % Remove padding from both ends
        startIdx = padLength + 1;
        endIdx = startIdx + size(audioData, 1) - 1;
        beamformedSignal = beamformedSignal(startIdx:min(endIdx, length(beamformedSignal)));
    else
    % Fallback if something went wrong
        beamformedSignal = beamformedSignal(1:min(size(audioData, 1), length(beamformedSignal)));
    end
end

%% Replace the estimateNoiseCovariance function with this improved version:

function Rnn = estimateNoiseCovariance(S, speechMask, fs, T, reg)
    [nFreqs, nFrames, nMics] = size(S);
    
    % Fix the frame alignment calculation
    frameRate = 1 / (T(2) - T(1));  % More accurate frame rate
    speechFrames = false(nFrames, 1);
    
    % Better time-to-frame mapping
    for i = 1:nFrames
        frameTime = T(i);
        sampleIdx = round(frameTime * fs);
        sampleIdx = max(1, min(length(speechMask), sampleIdx));
        speechFrames(i) = speechMask(sampleIdx);
    end
    
    noiseFrames = ~speechFrames;
    
    % Apply temporal smoothing with better kernel
    if length(noiseFrames) > 5
        noiseFrames = movmean(double(noiseFrames), 5) > 0.5;
    end
    
    Rnn = complex(zeros(nMics, nMics, nFreqs));
    
    % Visualization: Plot noise frame detection
    figure('Name', 'Noise Frame Detection');
    subplot(3,1,1);
    plot((1:length(speechMask))/fs, speechMask);
    title('Original Speech Mask');
    xlabel('Time (s)'); ylabel('Speech Activity');
    
    subplot(3,1,2);
    plot(T, speechFrames);
    title('Speech Frames (STFT domain)');
    xlabel('Time (s)'); ylabel('Speech Activity');
    
    subplot(3,1,3);
    plot(T, noiseFrames);
    title('Noise Frames (STFT domain)');
    xlabel('Time (s)'); ylabel('Noise Activity');
    
    noiseFrameCount = sum(noiseFrames);
    fprintf('    Noise frames: %d/%d (%.1f%%)\n', noiseFrameCount, nFrames, 100*noiseFrameCount/nFrames);
    
    if noiseFrameCount < max(5, 0.05 * nFrames)  % Need at least 5% noise frames
        fprintf('    WARNING: Insufficient noise frames! Using diagonal loading only\n');
        for k = 1:nFreqs
            Rnn(:,:,k) = eye(nMics) * reg * 100;  % Much higher regularization
        end
        return;
    end
    
    % Calculate covariance matrices with better numerical stability
    for k = 1:nFreqs
        noiseData = squeeze(S(k, noiseFrames, :));  % [noiseFrames x nMics]
        
        if size(noiseData, 1) >= 3  % Need at least 3 samples
            % Remove mean (important for covariance)
            noiseData = noiseData - mean(noiseData, 1);
            
            % Compute sample covariance
            R_sample = (noiseData' * noiseData) / (size(noiseData, 1) - 1);
            
            % Frequency-dependent regularization
            freq = (k-1) * fs / (2 * (nFreqs-1));
            freqReg = reg * (1 + 2 * exp(-freq/500));  % More regularization at low frequencies
            
            Rnn(:,:,k) = R_sample + freqReg * eye(nMics);
        else
            Rnn(:,:,k) = eye(nMics) * reg * 10;
        end
    end
    
    % Visualization: Plot covariance matrices at key frequencies
    visualizeNoiseCovariance(Rnn, fs, nFreqs);
end

%% Add this new visualization function:

function visualizeNoiseCovariance(Rnn, fs, nFreqs)
    % Visualize noise covariance matrices at different frequencies
    
    % Select key frequencies to visualize
    freqIndices = [2, round(nFreqs*0.1), round(nFreqs*0.3), round(nFreqs*0.5), round(nFreqs*0.8)];
    freqValues = (freqIndices-1) * fs / (2 * (nFreqs-1));
    
    figure('Name', 'Noise Covariance Matrices', 'Position', [100, 100, 1200, 800]);
    
    for idx = 1:length(freqIndices)
        k = freqIndices(idx);
        R = squeeze(Rnn(:,:,k));
        
        % Plot magnitude
        subplot(3, length(freqIndices), idx);
        imagesc(abs(R));
        colorbar;
        title(sprintf('|R| at %.0f Hz', freqValues(idx)));
        axis equal tight;
        
        % Plot phase
        subplot(3, length(freqIndices), idx + length(freqIndices));
        imagesc(angle(R));
        colorbar;
        title(sprintf('∠R at %.0f Hz', freqValues(idx)));
        axis equal tight;
        
        % Plot eigenvalues
        subplot(3, length(freqIndices), idx + 2*length(freqIndices));
        eigs = sort(real(eig(R)), 'descend');
        semilogy(eigs, 'o-');
        title(sprintf('Eigenvalues at %.0f Hz', freqValues(idx)));
        ylabel('Log Eigenvalue');
        grid on;
    end
    
    % Overall statistics
    figure('Name', 'Covariance Matrix Statistics');
    
    % Condition numbers across frequency
    condNumbers = zeros(nFreqs, 1);
    detValues = zeros(nFreqs, 1);
    
    for k = 1:nFreqs
        R = squeeze(Rnn(:,:,k));
        condNumbers(k) = cond(R);
        detValues(k) = real(det(R));
    end
    
    freqs = (0:nFreqs-1) * fs / (2 * (nFreqs-1));
    
    subplot(2,2,1);
    semilogy(freqs, condNumbers);
    title('Condition Number vs Frequency');
    xlabel('Frequency (Hz)'); ylabel('Condition Number');
    grid on;
    
    subplot(2,2,2);
    semilogy(freqs, detValues);
    title('Determinant vs Frequency');
    xlabel('Frequency (Hz)'); ylabel('Determinant');
    grid on;
    
    % Cross-correlation structure
    subplot(2,2,3);
    R_1kHz = squeeze(Rnn(:,:,round(nFreqs*0.25))); % ~1kHz
    R_normalized = R_1kHz ./ sqrt(diag(R_1kHz) * diag(R_1kHz)');
    imagesc(abs(R_normalized));
    colorbar;
    title('Normalized Correlation (1kHz)');
    axis equal tight;
    
    subplot(2,2,4);
    % Average diagonal loading across frequency
    diagLoading = zeros(nFreqs, 1);
    for k = 1:nFreqs
        R = squeeze(Rnn(:,:,k));
        diagLoading(k) = mean(diag(R)) - mean(R(~eye(size(R))));
    end
    plot(freqs, diagLoading);
    title('Diagonal Loading vs Frequency');
    xlabel('Frequency (Hz)'); ylabel('Diagonal Loading');
    grid on;
    
    % Print diagnostics
    fprintf('\n=== Noise Covariance Matrix Diagnostics ===\n');
    fprintf('Condition number range: %.2e to %.2e\n', min(condNumbers), max(condNumbers));
    fprintf('Mean condition number: %.2e\n', mean(condNumbers));
    fprintf('Frequencies with high condition (>1e6): %d/%d\n', sum(condNumbers > 1e6), nFreqs);
    
    % Check for potential issues
    if mean(condNumbers) > 1e8
        fprintf('WARNING: Very high condition numbers - matrix may be ill-conditioned!\n');
    end
    
    if sum(condNumbers > 1e10) > 0.1 * nFreqs
        fprintf('WARNING: Many frequencies have condition numbers > 1e10\n');
    end
    
    % Check diagonal dominance
    avgDiagDominance = 0;
    for k = 1:min(10, nFreqs)  % Check first 10 frequencies
        R = squeeze(Rnn(:,:,k));
        diagDom = mean(diag(R)) / mean(abs(R(~eye(size(R)))));
        avgDiagDominance = avgDiagDominance + diagDom;
    end
    avgDiagDominance = avgDiagDominance / min(10, nFreqs);
    
    fprintf('Average diagonal dominance: %.2f\n', avgDiagDominance);
    if avgDiagDominance < 2
        fprintf('WARNING: Low diagonal dominance - consider increasing regularization\n');
    end
end

%% Also replace the improvedVAD function to fix potential issues:

function speechMask = improvedVAD(signal, fs)
    % Improved VAD with better parameter tuning
    
    % STFT for spectral analysis
    windowLength = round(0.025 * fs);  % 25ms
    overlap = round(0.5 * windowLength);  % 50% overlap (was 75% - too much)
    
    [S, F, T] = stft(signal, fs, 'Window', hann(windowLength, 'periodic'), ...
                     'OverlapLength', overlap);
    
    % Compute spectral features
    spectralCentroid = computeSpectralCentroid(S, F);
    spectralEntropy = computeSpectralEntropy(abs(S));
    energy = sum(abs(S).^2, 1);
    
    % Add spectral rolloff feature
    spectralRolloff = computeSpectralRolloff(abs(S), F, 0.85);
    
    % Normalize features robustly
    spectralCentroid = normalizeFeature(spectralCentroid);
    spectralEntropy = normalizeFeature(spectralEntropy);
    energy = normalizeFeature(energy);
    spectralRolloff = normalizeFeature(spectralRolloff);
    
    % Improved feature combination
    vadScore = 0.4 * energy + 0.25 * spectralCentroid + 0.2 * spectralEntropy + 0.15 * spectralRolloff;
    
    % Use percentile-based adaptive threshold
    threshold = prctile(vadScore, 30);  % 30th percentile instead of 25th
    speechFrames = vadScore > threshold;
    
    % Improved temporal smoothing
    speechFrames = medfilt1(double(speechFrames), 5) > 0.5;
    % Additional morphological operations
    speechFrames = imopen(speechFrames, ones(3,1));  % Remove isolated speech frames
    speechFrames = imclose(speechFrames, ones(7,1)); % Fill small gaps
    
    % Convert frame decisions to sample mask with better alignment
    speechMask = false(length(signal), 1);
    samplesPerFrame = round((T(2) - T(1)) * fs);
    
    for i = 1:length(speechFrames)
        if speechFrames(i)
            startIdx = round((i-1) * samplesPerFrame) + 1;
            endIdx = min(startIdx + samplesPerFrame - 1, length(signal));
            if startIdx <= length(signal)
                speechMask(startIdx:endIdx) = true;
            end
        end
    end
    
    % Visualization of VAD
    figure('Name', 'VAD Analysis');
    subplot(3,1,1);
    t = (1:length(signal))/fs;
    plot(t, signal);
    hold on;
    plot(t, speechMask * max(abs(signal)), 'r', 'LineWidth', 2);
    title('Signal and Speech Mask');
    legend('Signal', 'Speech Mask');
    
    subplot(3,1,2);
    plot(T, vadScore);
    hold on;
    yline(threshold, 'r--', 'Threshold');
    title('VAD Score');
    
    subplot(3,1,3);
    plot(T, speechFrames);
    title('Final Speech Frames');
    xlabel('Time (s)');
end

function normalizedFeature = normalizeFeature(feature)
    % Robust feature normalization using median and MAD
    medianVal = median(feature);
    madVal = mad(feature, 1);
    if madVal < eps
        madVal = 1;
    end
    normalizedFeature = (feature - medianVal) / madVal;
end

function rolloff = computeSpectralRolloff(magnitude, F, threshold)
    % Compute spectral rolloff frequency
    totalEnergy = sum(magnitude, 1);
    cumulativeEnergy = cumsum(magnitude, 1);
    
    rolloff = zeros(1, size(magnitude, 2));
    for i = 1:size(magnitude, 2)
        if totalEnergy(i) > eps
            thresholdEnergy = threshold * totalEnergy(i);
            idx = find(cumulativeEnergy(:, i) >= thresholdEnergy, 1);
            if ~isempty(idx)
                rolloff(i) = F(idx);
            else
                rolloff(i) = F(end);
            end
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
    % Minimal post-processing for ASR - Whisper is robust to noise
    
    % Only essential processing:
    
    % 1. Remove DC offset
    signal = signal - mean(signal);
    
    % 2. Very gentle normalization (preserve dynamics for Whisper)
    maxAbs = max(abs(signal));
    if maxAbs > 0.95  % Only normalize if clipping risk
        signal = signal * (0.9 / maxAbs);
    end
    
    % 3. Minimal fade-in/out to prevent clicks (5ms only)
    fadeLength = round(0.005 * fs);
    if length(signal) > 2 * fadeLength
        signal(1:fadeLength) = signal(1:fadeLength) .* linspace(0, 1, fadeLength)';
        signal(end-fadeLength+1:end) = signal(end-fadeLength+1:end) .* linspace(1, 0, fadeLength)';
    end
    
    % That's it - no filtering, no spectral subtraction, no aggressive normalization
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
function signal = manualIFFTReconstruction(STFT, windowLength, overlap, nFFT)
    [~, nFrames] = size(STFT);
    hopLength = windowLength - overlap;
    outputLength = (nFrames - 1) * hopLength + windowLength;
    signal = zeros(outputLength, 1);
    window = hann(windowLength, 'periodic');
    
    for frameIdx = 1:nFrames
        % Ensure conjugate symmetry for real output
        frame_fft = STFT(:, frameIdx);
        if mod(nFFT, 2) == 0
            frame_fft(end) = real(frame_fft(end)); % Nyquist must be real
        end
        frame_fft(1) = real(frame_fft(1)); % DC must be real
        
        frameSignal = real(ifft(frame_fft, nFFT));
        frameSignal = frameSignal(1:windowLength) .* window;  % Apply window!
        
        startIdx = (frameIdx - 1) * hopLength + 1;
        endIdx = min(startIdx + windowLength - 1, length(signal));
        
        if startIdx <= length(signal)
            signal(startIdx:endIdx) = signal(startIdx:endIdx) + frameSignal(1:endIdx-startIdx+1);
        end
    end
end