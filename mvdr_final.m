% Streamlined MVDR Beamforming for Circular Array + Whisper ASR
% Optimized for 7-microphone circular array (6 outer + 1 center)
% Enhanced with frequency-domain MVDR beamforming using Signal Processing Toolbox

clc; clear; close all;

%% Configuration
inputDir = '/media/niklas/SSD2/ind_beamforming/';
outputDir = '/media/niklas/SSD2/whisper_beamforming/';

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
% Test mode configuration
TEST_MODE = true;  % Set to true for synthetic testing
if TEST_MODE
    % Test parameters
    testDuration = 3.0;        % 3 seconds
    testFrequency = 1000;      % 1000 Hz tone
    testAzimuth = 45;          % True DOA in degrees
    testSNR = 20;              % SNR in dB
    
    fprintf('=== SYNTHETIC TEST MODE ===\n');
    fprintf('Generating test signal: %.0f Hz from %.0f degrees\n', testFrequency, testAzimuth);
end
% Array geometry - circular array with center microphone
radius = 0.035;           % 3.5 cm radius
nMics = 7;                % 6 outer + 1 center
channels = [1, 2, 3, 4, 5, 6, 7];  % All 7 channels
targetFs = 16000;         % Whisper sample rate
c = 343;                  % Speed of sound (m/s)

% MVDR parameters
regularization = 1e-3;    % Diagonal loading factor
useFrequencyDomain = true; % Set to false to use original single-frequency method

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

if TEST_MODE
    % Generate single test case
    [audioData, trueAzimuth] = generateTestSignal(arrayPos, targetFs, testDuration, ...
                                                  testFrequency, testAzimuth, testSNR);
    
    fprintf('\n=== Processing Synthetic Test Signal ===\n');
    fprintf('True DOA: %.1f degrees\n', trueAzimuth);
    
    % Process the synthetic signal
    try
        % Estimate DOA
        fprintf('  Estimating DOA using %s...\n', doaMethod);
        estimatedDOA = estimateDOA(audioData, arrayPos, azimuthRange, freqRange, ...
                                 targetFs, c, doaMethod);
        fprintf('  Estimated DOA: %.1f degrees (Error: %.1f degrees)\n', ...
                estimatedDOA, abs(estimatedDOA - trueAzimuth));
        
        % Apply MVDR beamforming
        lookDirection = [estimatedDOA; 0];
        
        if useFrequencyDomain
            fprintf('  Using frequency-domain MVDR beamforming...\n');
            beamformedSignal = applyFrequencyDomainMVDR(audioData, arrayPos, lookDirection, ...
                                                      targetFs, c, regularization);
        else
            fprintf('  Using single-frequency MVDR beamforming...\n');
            beamformedSignal = applyMVDRBeamforming(audioData, arrayPos, lookDirection, ...
                                                  targetFs, c, regularization);
        end
        
        % Save outputs for analysis
        audiowrite(fullfile(outputDir, 'test_original_center.wav'), audioData(:,7), targetFs);
        audiowrite(fullfile(outputDir, 'test_beamformed.wav'), beamformedSignal, targetFs);
        
        % Compute and display metrics
        fprintf('\n=== Analysis Results ===\n');
        
        % Signal quality metrics
        originalSignal = audioData(:,7);  % Center microphone
        
        % SNR improvement
        % Estimate noise from first 0.1 seconds (assuming clean signal)
        noiseStart = round(0.05 * targetFs);
        noiseEnd = round(0.1 * targetFs);
        if noiseEnd < length(originalSignal)
            noiseEst = std(originalSignal(noiseStart:noiseEnd));
            signalEst = std(originalSignal);
            originalSNR = 20 * log10(signalEst / noiseEst);
            
            noiseEst_bf = std(beamformedSignal(noiseStart:noiseEnd));
            signalEst_bf = std(beamformedSignal);
            beamformedSNR = 20 * log10(signalEst_bf / noiseEst_bf);
            
            fprintf('Original SNR: %.1f dB\n', originalSNR);
            fprintf('Beamformed SNR: %.1f dB\n', beamformedSNR);
            fprintf('SNR Improvement: %.1f dB\n', beamformedSNR - originalSNR);
        end
        
        % Spectral analysis
       % Spectral analysis
        fprintf('\nPerforming spectral analysis...\n');
        
        % Create figure with better formatting
        fig = figure('Position', [100, 100, 1200, 800]);
        
        % Original vs beamformed spectrogram
        subplot(2,3,1);
        spectrogram(originalSignal, hann(512), 256, 1024, targetFs, 'yaxis');
        title('Original Signal (Center Mic)');
        colorbar;
        
        subplot(2,3,2);
        spectrogram(beamformedSignal, hann(512), 256, 1024, targetFs, 'yaxis');
        title('Beamformed Signal');
        colorbar;
        
        % Frequency domain comparison
        subplot(2,3,3);
        [pxx_orig, f] = pwelch(originalSignal, hann(512), 256, 1024, targetFs);
        [pxx_beam, ~] = pwelch(beamformedSignal, hann(512), 256, 1024, targetFs);
        
        semilogx(f, 10*log10(pxx_orig), 'b-', 'LineWidth', 1.5);
        hold on;
        semilogx(f, 10*log10(pxx_beam), 'r-', 'LineWidth', 1.5);
        % Highlight the test frequency
        xline(testFrequency, 'k--', 'LineWidth', 2);
        grid on;
        xlabel('Frequency (Hz)');
        ylabel('Power Spectral Density (dB/Hz)');
        title('PSD Comparison');
        legend('Original', 'Beamformed', sprintf('Test Freq (%d Hz)', testFrequency), 'Location', 'best');
        xlim([100, 8000]);
        
        % Time domain comparison (zoomed to show waveform detail)
        subplot(2,3,4);
        t = (0:length(originalSignal)-1) / targetFs;
        % Show only first 0.1 seconds for detail
        timeRange = t <= 0.1;
        plot(t(timeRange), originalSignal(timeRange), 'b-', 'LineWidth', 1);
        hold on;
        plot(t(timeRange), beamformedSignal(timeRange), 'r-', 'LineWidth', 1);
        grid on;
        xlabel('Time (s)');
        ylabel('Amplitude');
        title('Time Domain Comparison (First 0.1s)');
        legend('Original', 'Beamformed', 'Location', 'best');
        
        % Difference signal analysis
        subplot(2,3,5);
        diffSignal = beamformedSignal - originalSignal;
        plot(t, diffSignal, 'g-', 'LineWidth', 1);
        grid on;
        xlabel('Time (s)');
        ylabel('Amplitude');
        title('Difference Signal (Beamformed - Original)');
        
        % Artifact detection: High-frequency content analysis
        subplot(2,3,6);
        % Check for harmonics and artifacts
        [pxx_diff, f_diff] = pwelch(diffSignal, hann(512), 256, 1024, targetFs);
        semilogx(f_diff, 10*log10(max(pxx_diff, eps)), 'g-', 'LineWidth', 1.5);
        grid on;
        xlabel('Frequency (Hz)');
        ylabel('Power Spectral Density (dB/Hz)');
        title('Difference Signal PSD (Artifacts)');
        xlim([100, 8000]);
        
        % Save in multiple formats
        saveas(fig, fullfile(outputDir, 'test_analysis.png'));
        saveas(fig, fullfile(outputDir, 'test_analysis.pdf'));
        
        % Additional numerical analysis
        fprintf('\n=== Detailed Artifact Analysis ===\n');
        
        % Check for harmonic distortion
        harmonics = [2, 3, 4, 5] * testFrequency;
        [pxx_beam_detailed, f_detailed] = pwelch(beamformedSignal, hann(2048), 1024, 4096, targetFs);
        
        fprintf('Harmonic analysis:\n');
        for i = 1:length(harmonics)
            if harmonics(i) < targetFs/2
                [~, harmonic_idx] = min(abs(f_detailed - harmonics(i)));
                harmonic_power = 10*log10(pxx_beam_detailed(harmonic_idx));
                
                % Compare to fundamental
                [~, fund_idx] = min(abs(f_detailed - testFrequency));
                fund_power = 10*log10(pxx_beam_detailed(fund_idx));
                
                fprintf('  %d Hz harmonic: %.1f dB (%.1f dB below fundamental)\n', ...
                        harmonics(i), harmonic_power, fund_power - harmonic_power);
            end
        end
        
        % RMS comparison
        rms_orig = sqrt(mean(originalSignal.^2));
        rms_beam = sqrt(mean(beamformedSignal.^2));
        rms_diff = sqrt(mean(diffSignal.^2));
        
        fprintf('\nRMS Analysis:\n');
        fprintf('  Original RMS: %.6f\n', rms_orig);
        fprintf('  Beamformed RMS: %.6f\n', rms_beam);
        fprintf('  Difference RMS: %.6f (%.1f%% of original)\n', ...
                rms_diff, 100 * rms_diff / rms_orig);
        
        % Check for clipping or saturation
        max_orig = max(abs(originalSignal));
        max_beam = max(abs(beamformedSignal));
        
        fprintf('\nAmplitude Analysis:\n');
        fprintf('  Original peak: %.6f\n', max_orig);
        fprintf('  Beamformed peak: %.6f\n', max_beam);
        
        if max_beam > 0.95
            fprintf('  WARNING: Possible clipping detected!\n');
        end
        
        % THD calculation (Total Harmonic Distortion)
        fund_power_linear = pxx_beam_detailed(fund_idx);
        harmonic_power_linear = 0;
        for i = 1:length(harmonics)
            if harmonics(i) < targetFs/2
                [~, harmonic_idx] = min(abs(f_detailed - harmonics(i)));
                harmonic_power_linear = harmonic_power_linear + pxx_beam_detailed(harmonic_idx);
            end
        end
        
        thd_percent = 100 * sqrt(harmonic_power_linear / fund_power_linear);
        fprintf('  Total Harmonic Distortion: %.3f%%\n', thd_percent);
        
        if thd_percent > 1.0
            fprintf('  WARNING: High THD detected - check for processing artifacts!\n');
        end
        
    catch ME
        fprintf('Error in test: %s\n', ME.message);
    end
    
else
    % Original processing loop for real data
    for segIdx = 1:length(segmentIds)
        % ... (keep your existing processing loop unchanged)
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
    windowLength = round(0.032 * fs);  % 32ms window
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
    beamformedSignal = istft(beamformedSTFT, fs, 'Window', hann(windowLength, 'periodic'), ...
                        'OverlapLength', overlap, 'FFTLength', nFFT, ...
                        'ConjugateSymmetric', true);
    
    % Ensure output length matches input
    if length(beamformedSignal) > nSamples
        beamformedSignal = beamformedSignal(1:nSamples);
    elseif length(beamformedSignal) < nSamples
        beamformedSignal = [beamformedSignal; zeros(nSamples - length(beamformedSignal), 1)];
    end
  
    
    % Apply post-processing
    beamformedSignal = postProcessSignal(beamformedSignal, fs);
end

function Rnn = estimateNoiseCovariance(S, speechMask, fs, T, reg)
    % Estimate noise covariance matrices using robust statistics
    [nFreqs, nFrames, nMics] = size(S);
    
    % Convert speech mask to frame indices
    frameRate = 1 / (T(2) - T(1));  % Frames per second
    speechFrames = false(nFrames, 1);
    
    for i = 1:nFrames
        timeIdx = round(T(i) * fs);
        if timeIdx > 0 && timeIdx <= length(speechMask)
            speechFrames(i) = speechMask(timeIdx);
        end
    end
    
    noiseFrames = ~speechFrames;
    
    % Initialize covariance matrices
    Rnn = complex(zeros(nMics, nMics, nFreqs));
    
    if sum(noiseFrames) < 5  % Need minimum frames
        fprintf('    Insufficient noise frames, using identity covariance\n');
        for k = 1:nFreqs
            Rnn(:,:,k) = eye(nMics) * reg;
        end
        return;
    end
    
    % Estimate covariance for each frequency bin
    for k = 1:nFreqs
        noiseData = squeeze(S(k, noiseFrames, :));  % [nNoiseFrames x nMics]
        
        if size(noiseData, 1) > 1
            % Robust covariance estimation
            Rnn(:,:,k) = (noiseData' * noiseData) / size(noiseData, 1);
        else
            Rnn(:,:,k) = eye(nMics) * reg;
        end
        
        % Add diagonal loading
        Rnn(:,:,k) = Rnn(:,:,k) + reg * eye(nMics);
    end
end
% Step-by-step analysis to isolate artifacts
        analyzeProcessingSteps(audioData, arrayPos, lookDirection, targetFs, c, regularization, outputDir);
function speechMask = improvedVAD(signal, fs)
    % Improved VAD using spectral features and Signal Processing Toolbox
    
    % STFT for spectral analysis
    windowLength = round(0.025 * fs);  % 25ms
    overlap = round(0.01 * fs);        % 10ms
    
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
    vadScore = 0.4 * energy + 0.3 * spectralCentroid + 0.3 * spectralEntropy;
    
    % Adaptive threshold
    threshold = prctile(vadScore, 35);
    speechFrames = vadScore > threshold;
    
    % Smooth decisions with median filter
    speechFrames = medfilt1(double(speechFrames), 5) > 0.5;
    
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
    % Post-processing using Signal Processing Toolbox
    
    % Normalize output
    rmsLevel = sqrt(mean(signal.^2));
    if rmsLevel > 0
        signal = signal * (0.1 / rmsLevel);  % Target RMS level
    end
    % Multi-stage filtering
    % 1. High-pass filter to remove DC and low-frequency noise
    
    % 2. Notch filter for 50Hz/60Hz hum (if needed)

    % 3. Light anti-aliasing filter

    
    % 4. Dynamic range compression (optional)
    % signal = compressDynamicRange(signal, 0.7);  % Uncomment if needed
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
    windowLength = round(0.032 * fs);
    overlap = round(0.75 * windowLength);
    
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
% Streamlined MVDR Beamforming for Circular Array + Whisper ASR
% Optimized for 7-microphone circular array (6 outer + 1 center)
% Enhanced with frequency-domain MVDR beamforming using Signal Processing Toolbox
function [testAudioData, trueAzimuth] = generateTestSignal(arrayPos, fs, duration, freq, azimuth, snr_db)
    % Generate synthetic test signal with known DOA
    nMics = size(arrayPos, 2);
    nSamples = round(duration * fs);
    c = 343; % Speed of sound
    
    % Generate clean 1000Hz tone
    t = (0:nSamples-1)' / fs;
    cleanSignal = sin(2 * pi * freq * t);
    
    % Compute steering vector for true direction
    lookDir = [cosd(azimuth); sind(azimuth); 0];
    refPos = arrayPos(:, 7); % Center mic as reference
    
    % Initialize multichannel data
    testAudioData = zeros(nSamples, nMics);
    
    % Apply delays and add to each microphone
    for i = 1:nMics
        deltaPos = arrayPos(:, i) - refPos;
        timeDelay = dot(deltaPos, lookDir) / c;
        
        % Fractional delay using interpolation
        delaySamples = timeDelay * fs;
        if abs(delaySamples) > 0.1
            delayedSignal = delaySignal(cleanSignal, delaySamples, fs);
        else
            delayedSignal = cleanSignal;
        end
        
        testAudioData(:, i) = delayedSignal;
    end
    
    % Add uncorrelated noise to each channel
    if snr_db < inf
        signalPower = mean(testAudioData.^2, 1);
        noisePower = signalPower / (10^(snr_db/10));
        
        for i = 1:nMics
            noise = sqrt(noisePower(i)) * randn(nSamples, 1);
            testAudioData(:, i) = testAudioData(:, i) + noise;
        end
    end
    
    trueAzimuth = azimuth;
end

function delayedSignal = delaySignal(signal, delaySamples, fs)
    % Apply fractional delay using sinc interpolation
    if abs(delaySamples) < 1e-6
        delayedSignal = signal;
        return;
    end
    
    % Use simple linear interpolation for small delays
    if abs(delaySamples) < 10
        delayedSignal = interp1((1:length(signal))', signal, ...
                               (1:length(signal))' - delaySamples, ...
                               'linear', 'extrap');
    else
        % For larger delays, use time-domain shifting
        integerDelay = round(delaySamples);
        fractionalDelay = delaySamples - integerDelay;
        
        if integerDelay > 0
            delayedSignal = [zeros(integerDelay, 1); signal(1:end-integerDelay)];
        elseif integerDelay < 0
            delayedSignal = [signal(-integerDelay+1:end); zeros(-integerDelay, 1)];
        else
            delayedSignal = signal;
        end
        
        % Apply fractional delay if needed
        if abs(fractionalDelay) > 1e-6
            delayedSignal = interp1((1:length(delayedSignal))', delayedSignal, ...
                                   (1:length(delayedSignal))' - fractionalDelay, ...
                                   'linear', 'extrap');
        end
    end
end
function analyzeProcessingSteps(audioData, arrayPos, lookDirection, fs, c, regularization, outputDir)
    % Step-by-step analysis to isolate artifact sources
    fprintf('\n=== Step-by-Step Processing Analysis ===\n');
    
    % Step 1: Just delay-and-sum (no MVDR)
    fprintf('Step 1: Simple delay-and-sum beamforming...\n');
    delayAndSumSignal = applyDelayAndSum(audioData, arrayPos, lookDirection, fs, c);
    audiowrite(fullfile(outputDir, 'step1_delay_sum.wav'), delayAndSumSignal, fs);
    
    % Step 2: MVDR without post-processing
    fprintf('Step 2: MVDR without post-processing...\n');
    mvdrRawSignal = applyMVDRRaw(audioData, arrayPos, lookDirection, fs, c, regularization);
    audiowrite(fullfile(outputDir, 'step2_mvdr_raw.wav'), mvdrRawSignal, fs);
    
    % Step 3: Only post-processing effects
    fprintf('Step 3: Post-processing only...\n');
    postProcOnlySignal = postProcessSignal(audioData(:,7), fs);  % Center mic + post-proc
    audiowrite(fullfile(outputDir, 'step3_postproc_only.wav'), postProcOnlySignal, fs);
    
    fprintf('Step-by-step analysis files saved for comparison.\n');
end

function beamformedSignal = applyDelayAndSum(audioData, arrayPos, lookDirection, fs, c)
    % Simple delay-and-sum beamforming for comparison
    [nSamples, nMics] = size(audioData);
    
    % Compute delays
    azimuth = lookDirection(1) * pi/180;
    lookDir = [cos(azimuth); sin(azimuth); 0];
    refPos = arrayPos(:, 7);  % Center mic as reference
    
    alignedSignals = zeros(size(audioData));
    
    for i = 1:nMics
        deltaPos = arrayPos(:, i) - refPos;
        timeDelay = dot(deltaPos, lookDir) / c;
        delaySamples = timeDelay * fs;
        
        if abs(delaySamples) > 0.1
            alignedSignals(:, i) = delaySignal(audioData(:, i), -delaySamples, fs);
        else
            alignedSignals(:, i) = audioData(:, i);
        end
    end
    
    beamformedSignal = mean(alignedSignals, 2);
end

function beamformedSignal = applyMVDRRaw(audioData, arrayPos, lookDirection, fs, c, reg)
    % MVDR without any post-processing
    [nSamples, nMics] = size(audioData);
    
    % Simple VAD
    speechMask = improvedVAD(audioData(:,7), fs);
    noiseIndices = find(~speechMask);
    
    if length(noiseIndices) < round(0.025 * fs)
        Rnn = eye(nMics) * reg;
    else
        noiseData = audioData(noiseIndices, :);
        Rnn = robustCov(noiseData) + reg * eye(nMics);
    end
    
    % Compute steering vector and weights
    freq = 1000;
    steeringVector = computeSteeringVector(arrayPos, lookDirection, freq, c);
    
    try
        w = (Rnn \ steeringVector) / (steeringVector' / Rnn * steeringVector);
    catch
        w = steeringVector / (steeringVector' * steeringVector);
    end
    
    % Apply beamforming WITHOUT post-processing
    beamformedSignal = real(audioData * conj(w));
end
clc; clear; close all;

