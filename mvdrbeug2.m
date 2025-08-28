set(groot, 'DefaultFigureRenderer', 'painters');  % Try different renderer


clc; clear; close all;

outputDir = './mvdr_debug_results/';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

function arrayPos = getCircularArrayPositions(radius, nMics)
  
    arrayPos = zeros(3, nMics);
    
  
    for i = 1:6
        angle = (i-1) * 2*pi / 6;  % 6 outer mics, starting at 0°
        arrayPos(:, i) = [radius * cos(angle); radius * sin(angle); 0];
    end
    
    arrayPos(:, 7) = [0; 0; 0];
end

function [sourceSignal, micSignals, interferenceSignal] = generateTestSignalWithInterference(...
    arrayPos, sourceAngle, sourceFreq, interferenceAngle, interferenceFreq, fs, duration, snr_dB)

    
    nMics = size(arrayPos, 2);
    nSamples = round(duration * fs);
    t = (0:nSamples-1)' / fs;
    c = 343; % Speed of sound

    sourceSignal = sin(2*pi*sourceFreq*t);

    if ~isempty(interferenceAngle) && ~isempty(interferenceFreq)
        interferenceSignal = 0.8 * sin(2*pi*interferenceFreq*t + pi/4); % Phase shift for realism
    else
        interferenceSignal = zeros(size(sourceSignal));
    end
    sourceAngle_rad = deg2rad(sourceAngle);
    sourceDir = [cos(sourceAngle_rad); sin(sourceAngle_rad); 0];
   
    micSignals = zeros(nSamples, nMics);
    
    for i = 1:nMics
        % Calculate time delay from source to microphone
        distance_diff = dot(arrayPos(:, i), sourceDir);
        delay_samples = round(distance_diff * fs / c);

        if delay_samples >= 0
            delayed_source = [zeros(delay_samples, 1); sourceSignal(1:end-delay_samples)];
        else
            delayed_source = [sourceSignal(-delay_samples+1:end); zeros(-delay_samples, 1)];
        end
        
        micSignals(:, i) = delayed_source;
    end
    
    % Add interference if specified
    if ~isempty(interferenceAngle) && ~isempty(interferenceFreq)
        interferenceAngle_rad = deg2rad(interferenceAngle);
        interferenceDir = [cos(interferenceAngle_rad); sin(interferenceAngle_rad); 0];
        
        for i = 1:nMics
            distance_diff = dot(arrayPos(:, i), interferenceDir);
            delay_samples = round(distance_diff * fs / c);
            
            if delay_samples >= 0
                delayed_interference = [zeros(delay_samples, 1); interferenceSignal(1:end-delay_samples)];
            else
                delayed_interference = [interferenceSignal(-delay_samples+1:end); zeros(-delay_samples, 1)];
            end
            
            micSignals(:, i) = micSignals(:, i) + delayed_interference;
        end
    end

    signal_power = mean(sourceSignal.^2);
    noise_power = signal_power / (10^(snr_dB/10));
    
    for i = 1:nMics
        noise = sqrt(noise_power) * randn(nSamples, 1);
        micSignals(:, i) = micSignals(:, i) + noise;
    end
end

function beamformedSignal = applyDASBeamforming(micSignals, arrayPos, beamAngle, fs, c)
    
    [nSamples, nMics] = size(micSignals);
    beamAngle_rad = deg2rad(beamAngle);
    beamDir = [cos(beamAngle_rad); sin(beamAngle_rad); 0];
    
    delays = zeros(nMics, 1);
    for i = 1:nMics
        distance_diff = dot(arrayPos(:, i), beamDir);
        delays(i) = round(distance_diff * fs / c);
    end
    
    alignedSignals = zeros(nSamples, nMics);
    max_delay = max(abs(delays));
    
    for i = 1:nMics
        delay = delays(i) - min(delays); % Make all delays positive
        if delay >= 0
            alignedSignals(1:nSamples-delay, i) = micSignals(delay+1:nSamples, i);
        else
            alignedSignals(-delay+1:nSamples, i) = micSignals(1:nSamples+delay, i);
        end
    end
    
    % Sum aligned signals
    beamformedSignal = mean(alignedSignals, 2);
end

function beamformedSignal = applyMVDRBeamforming(micSignals, arrayPos, beamAngle, fs, c, blockSize, overlap, diagonal_loading)
    % Apply MVDR beamforming in frequency domain
    
    [nSamples, nMics] = size(micSignals);
    hopSize = round(blockSize * (1 - overlap));
    nBlocks = floor((nSamples - blockSize) / hopSize) + 1;
    nFreqs = blockSize / 2 + 1;
    
    % Initialize output
    beamformedSignal = zeros(nSamples, 1);
    normalization = zeros(nSamples, 1);
    
    % Window function
    window = hann(blockSize);
    
    % Process each block
    for blockIdx = 1:nBlocks
        startIdx = (blockIdx - 1) * hopSize + 1;
        endIdx = min(startIdx + blockSize - 1, nSamples);
        blockLength = endIdx - startIdx + 1;
        
        if blockLength < blockSize
            continue; % Skip incomplete blocks
        end
        
        % Extract windowed block
        micBlock = micSignals(startIdx:endIdx, :);
        windowedBlock = micBlock .* repmat(window, 1, nMics);
        
        % FFT
        micBlockFFT = fft(windowedBlock, blockSize);
        micBlockFFT = micBlockFFT(1:nFreqs, :); % Keep positive frequencies only
        
        % Process each frequency bin
        beamformedBlockFFT = zeros(nFreqs, 1);
        
        for freqIdx = 1:nFreqs
            freq = (freqIdx - 1) * fs / blockSize;
            
            if freq < 100 % Skip very low frequencies
                beamformedBlockFFT(freqIdx) = mean(micBlockFFT(freqIdx, :));
                continue;
            end
            
            % Get frequency domain signals
            X = micBlockFFT(freqIdx, :).'; % Column vector
            
            % Compute covariance matrix
            R = X * X' + diagonal_loading * eye(nMics);
            
            % Compute steering vector
            a = computeSteeringVector(arrayPos, beamAngle, freq, fs, c);
            
            % Compute MVDR weights
            w = computeMVDRWeights(R, a, diagonal_loading);
            
            % Apply beamforming
            beamformedBlockFFT(freqIdx) = w' * X;
        end
        
        % Reconstruct full spectrum (mirror for negative frequencies)
        fullSpectrum = [beamformedBlockFFT; conj(beamformedBlockFFT(end-1:-1:2))];
        
        % IFFT back to time domain
        beamformedBlock = real(ifft(fullSpectrum));
        beamformedBlock = beamformedBlock .* window;
        
        % Overlap-add
        beamformedSignal(startIdx:endIdx) = beamformedSignal(startIdx:endIdx) + beamformedBlock;
        normalization(startIdx:endIdx) = normalization(startIdx:endIdx) + window;
    end
    
    % Normalize overlap-add result
    validIdx = normalization > 0.01;
    beamformedSignal(validIdx) = beamformedSignal(validIdx) ./ normalization(validIdx);
end

function a = computeSteeringVector(arrayPos, beamAngle, freq, fs, c)
    % Compute steering vector for given beam angle and frequency
    
    nMics = size(arrayPos, 2);
    beamAngle_rad = deg2rad(beamAngle);
    beamDir = [cos(beamAngle_rad); sin(beamAngle_rad); 0];
    
    a = zeros(nMics, 1);
    k = 2 * pi * freq / c; % Wavenumber
    
    for i = 1:nMics
        phase_shift = k * dot(arrayPos(:, i), beamDir);
        a(i) = exp(-1j * phase_shift);
    end
end

function w = computeMVDRWeights(R, a, diagonal_loading)
    % Compute MVDR beamforming weights
    
    % Add diagonal loading for numerical stability
    R_reg = R + diagonal_loading * eye(size(R, 1));
    
    try
        % MVDR formula: w = (R^-1 * a) / (a^H * R^-1 * a)
        R_inv_a = R_reg \ a;
        w = R_inv_a / (a' * R_inv_a);
    catch
        % Fallback to uniform weights if matrix inversion fails
        w = a / norm(a);
    end
end

function R = computeCovarianceMatrix(micSignals, blockSize, overlap)
    % Compute sample covariance matrix in frequency domain
    
    [nSamples, nMics] = size(micSignals);
    hopSize = round(blockSize * (1 - overlap));
    nBlocks = floor((nSamples - blockSize) / hopSize) + 1;
    nFreqs = blockSize / 2 + 1;
    
    % Initialize covariance matrices
    R = zeros(nMics, nMics, nFreqs);
    window = hann(blockSize);
    
    for blockIdx = 1:nBlocks
        startIdx = (blockIdx - 1) * hopSize + 1;
        endIdx = min(startIdx + blockSize - 1, nSamples);
        blockLength = endIdx - startIdx + 1;
        
        if blockLength < blockSize
            continue;
        end
        
        % Extract windowed block
        micBlock = micSignals(startIdx:endIdx, :);
        windowedBlock = micBlock .* repmat(window, 1, nMics);
        
        % FFT
        micBlockFFT = fft(windowedBlock, blockSize);
        micBlockFFT = micBlockFFT(1:nFreqs, :);
        
        % Accumulate covariance for each frequency
        for freqIdx = 1:nFreqs
            X = micBlockFFT(freqIdx, :).'; % Column vector
            R(:, :, freqIdx) = R(:, :, freqIdx) + X * X';
        end
    end
    
    % Average over blocks
    R = R / nBlocks;
end

function sinr_dB = measureSINR(signal, targetFreq, interferenceFreq, fs)
  
    [psd, freqs] = pwelch(signal, [], [], [], fs);
    

    [~, targetBin] = min(abs(freqs - targetFreq));
    
    if ~isempty(interferenceFreq)
        [~, interferenceBin] = min(abs(freqs - interferenceFreq));
        
  
        targetRange = max(1, targetBin-2):min(length(freqs), targetBin+2);
        signalPower = mean(psd(targetRange));
        

        interferenceRange = max(1, interferenceBin-2):min(length(freqs), interferenceBin+2);
        interferencePower = mean(psd(interferenceRange));
        
       
        excludeIdx = [targetRange, interferenceRange];
        noiseIdx = setdiff(1:length(psd), excludeIdx);
        noisePower = mean(psd(noiseIdx));
        
        sinr_dB = 10 * log10(signalPower / (interferencePower + noisePower));
    else
        % Only signal and noise
        targetRange = max(1, targetBin-2):min(length(freqs), targetBin+2);
        signalPower = mean(psd(targetRange));
        
        noiseIdx = setdiff(1:length(psd), targetRange);
        noisePower = mean(psd(noiseIdx));
        
        sinr_dB = 10 * log10(signalPower / noisePower);
    end
end

function power = measureSignalPower(signal, freq, fs)
    % Measure signal power at specific frequency
    
    [psd, freqs] = pwelch(signal, [], [], [], fs);
    [~, freqBin] = min(abs(freqs - freq));
    freqRange = max(1, freqBin-1):min(length(freqs), freqBin+1);
    power = mean(psd(freqRange));
end

function [f, P] = computeSpectrum(signal, fs)
    % Compute power spectrum of signal
    
    [P, f] = pwelch(signal, [], [], [], fs);
end

% Test parameters
fs = 16000;                % Sample rate (Hz)
duration = 3.0;            % Signal duration (seconds) - longer for covariance estimation
testFreq = 1000;           % Test frequency (Hz)
snr_dB = 20;               % Signal-to-noise ratio (dB)
interferenceFreq = 1500;   % Interference frequency (Hz)
interferenceAngle = 180;   % Interference direction (degrees)

% Array geometry - same as DAS script
radius = 0.035;            % 3.5 cm radius
nMics = 7;                 % 6 outer + 1 center
c = 343;                   % Speed of sound (m/s)

% MVDR specific parameters
blockSize = 2048;          % Block size for frequency domain processing
overlap = 0.5;             % Overlap factor
diagonal_loading = 1e-3;   % Diagonal loading factor for matrix regularization

% Test scenarios
sourceAngles = [0, 45, 90, 135, 180, 225, 270, 315]; % Different source directions
beamAngles = [0, 45, 90, 135, 180, 225, 270, 315];   % Different beam directions
interferenceAngles = [90, 135, 180, 225, 270];       % Interference directions


%% Setup array geometry
arrayPos = getCircularArrayPositions(radius, nMics);

% Display array geometry
fprintf('\nArray geometry:\n');
for i = 1:nMics
    fprintf('Mic %d: (%.3f, %.3f, %.3f) m\n', i, arrayPos(1,i), arrayPos(2,i), arrayPos(3,i));
end

%% Test 1: Basic MVDR vs DAS Comparison
fprintf('\n=== Test 1: MVDR vs DAS Comparison ===\n');

sourceAngle = 0;  % Source at 0 degrees (front)
beamAngle = 0;    % Beam pointing at 0 degrees

[testSignal, micSignals, interferenceSignal] = generateTestSignalWithInterference(...
    arrayPos, sourceAngle, testFreq, interferenceAngle, interferenceFreq, fs, duration, snr_dB);

beamformedSignal_DAS = applyDASBeamforming(micSignals, arrayPos, beamAngle, fs, c);
beamformedSignal_MVDR = applyMVDRBeamforming(micSignals, arrayPos, beamAngle, fs, c, ...
    blockSize, overlap, diagonal_loading);

figure('Position', [100, 100, 1400, 1000]);
set(gcf, 'Color', 'white');

subplot(3,3,1);
plot((0:length(testSignal)-1)/fs, testSignal, 'b-', 'LineWidth', 1.5);
title('Original Source Signal', 'FontSize', 12);
xlabel('Time (s)'); ylabel('Amplitude');
grid on; axis tight;

subplot(3,3,2);
plot((0:length(interferenceSignal)-1)/fs, interferenceSignal, 'r-', 'LineWidth', 1.5);
title('Interference Signal', 'FontSize', 12);
xlabel('Time (s)'); ylabel('Amplitude');
grid on; axis tight;

subplot(3,3,3);
plot((0:size(micSignals,1)-1)/fs, micSignals(:,1), 'g-', 'LineWidth', 1.2);
title('Mixed Signal at Mic 1', 'FontSize', 12);
xlabel('Time (s)'); ylabel('Amplitude');
grid on; axis tight;

subplot(3,3,4);
plot((0:length(beamformedSignal_DAS)-1)/fs, beamformedSignal_DAS, 'k-', 'LineWidth', 1.5);
title('DAS Beamformed Signal', 'FontSize', 12);
xlabel('Time (s)'); ylabel('Amplitude');
grid on; axis tight;

subplot(3,3,5);
plot((0:length(beamformedSignal_MVDR)-1)/fs, beamformedSignal_MVDR, 'm-', 'LineWidth', 1.5);
title('MVDR Beamformed Signal', 'FontSize', 12);
xlabel('Time (s)'); ylabel('Amplitude');
grid on; axis tight;

% Frequency analysis
subplot(3,3,6);
[f, P_orig] = computeSpectrum(testSignal, fs);
plot(f, 10*log10(P_orig), 'b-', 'LineWidth', 1.5);
hold on;
[f, P_int] = computeSpectrum(interferenceSignal, fs);
plot(f, 10*log10(P_int), 'r-', 'LineWidth', 1.5);
title('Original Signals Spectrum', 'FontSize', 12);
xlabel('Frequency (Hz)'); ylabel('Power (dB)');
legend('Source', 'Interference', 'Location', 'best');
grid on; xlim([0, 3000]); axis tight;

subplot(3,3,7);
[f, P_mic] = computeSpectrum(micSignals(:,1), fs);
plot(f, 10*log10(P_mic), 'g-', 'LineWidth', 1.5);
title('Mixed Signal Spectrum', 'FontSize', 12);
xlabel('Frequency (Hz)'); ylabel('Power (dB)');
grid on; xlim([0, 3000]); axis tight;

subplot(3,3,8);
[f, P_das] = computeSpectrum(beamformedSignal_DAS, fs);
plot(f, 10*log10(P_das), 'k-', 'LineWidth', 1.5);
title('DAS Beamformed Spectrum', 'FontSize', 12);
xlabel('Frequency (Hz)'); ylabel('Power (dB)');
grid on; xlim([0, 3000]); axis tight;

subplot(3,3,9);
[f, P_mvdr] = computeSpectrum(beamformedSignal_MVDR, fs);
plot(f, 10*log10(P_mvdr), 'm-', 'LineWidth', 1.5);
title('MVDR Beamformed Spectrum', 'FontSize', 12);
xlabel('Frequency (Hz)'); ylabel('Power (dB)');
grid on; xlim([0, 3000]); axis tight;

sgtitle(sprintf('MVDR vs DAS - Source at %d°, Interference at %d°', sourceAngle, interferenceAngle), 'FontSize', 14);
drawnow;
saveas(gcf, fullfile(outputDir, 'test1_mvdr_vs_das.png'));
fprintf('Saved: test1_mvdr_vs_das.png\n');

fprintf('\n=== Test 2: Interference Suppression Analysis ===\n');

sourceAngle = 0;  % Fixed source direction
beamAngle = 0;    % Fixed beam direction

sinr_input = zeros(size(interferenceAngles));
sinr_das = zeros(size(interferenceAngles));
sinr_mvdr = zeros(size(interferenceAngles));

for i = 1:length(interferenceAngles)
    [testSignal, micSignals, interferenceSignal] = generateTestSignalWithInterference(...
        arrayPos, sourceAngle, testFreq, interferenceAngles(i), interferenceFreq, fs, duration, snr_dB);
    
    beamformed_DAS = applyDASBeamforming(micSignals, arrayPos, beamAngle, fs, c);
    beamformed_MVDR = applyMVDRBeamforming(micSignals, arrayPos, beamAngle, fs, c, ...
        blockSize, overlap, diagonal_loading);
    
    % Measure SINR (Signal-to-Interference-plus-Noise Ratio)
    sinr_input(i) = measureSINR(micSignals(:,1), testFreq, interferenceFreq, fs);
    sinr_das(i) = measureSINR(beamformed_DAS, testFreq, interferenceFreq, fs);
    sinr_mvdr(i) = measureSINR(beamformed_MVDR, testFreq, interferenceFreq, fs);
end

% Plot interference suppression results
figure('Position', [200, 200, 1200, 800]);
set(gcf, 'Color', 'white');

subplot(2,3,1);
plot(interferenceAngles, sinr_input, 'g-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(interferenceAngles, sinr_das, 'k-s', 'LineWidth', 2, 'MarkerSize', 8);
plot(interferenceAngles, sinr_mvdr, 'm-^', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Interference Angle (degrees)'); ylabel('SINR (dB)');
title('SINR vs Interference Direction');
legend('Input', 'DAS', 'MVDR', 'Location', 'best');
grid on;

subplot(2,3,2);
sinr_improvement_das = sinr_das - sinr_input;
sinr_improvement_mvdr = sinr_mvdr - sinr_input;
bar(interferenceAngles, [sinr_improvement_das; sinr_improvement_mvdr]', 'grouped');
xlabel('Interference Angle (degrees)'); ylabel('SINR Improvement (dB)');
title('SINR Improvement Comparison');
legend('DAS', 'MVDR', 'Location', 'best');
grid on;

subplot(2,3,3);
interference_suppression_das = sinr_das - sinr_input;
interference_suppression_mvdr = sinr_mvdr - sinr_input;
plot(interferenceAngles, interference_suppression_das, 'k-s', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(interferenceAngles, interference_suppression_mvdr, 'm-^', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Interference Angle (degrees)'); ylabel('Interference Suppression (dB)');
title('Interference Suppression Performance');
legend('DAS', 'MVDR', 'Location', 'best');
grid on;

% Directivity patterns with interference
subplot(2,3,[4,5,6]);
% Test directivity with strong interference
[~, micSignals_with_int, ~] = generateTestSignalWithInterference(...
    arrayPos, 0, testFreq, 180, interferenceFreq, fs, duration, snr_dB);

beamPowers_das = zeros(size(beamAngles));
beamPowers_mvdr = zeros(size(beamAngles));

for i = 1:length(beamAngles)
    beamformed_das = applyDASBeamforming(micSignals_with_int, arrayPos, beamAngles(i), fs, c);
    beamformed_mvdr = applyMVDRBeamforming(micSignals_with_int, arrayPos, beamAngles(i), fs, c, ...
        blockSize, overlap, diagonal_loading);
    
    beamPowers_das(i) = mean(beamformed_das.^2);
    beamPowers_mvdr(i) = mean(beamformed_mvdr.^2);
end

% Normalize
beamPowers_das_dB = 10*log10(beamPowers_das) - max(10*log10(beamPowers_das));
beamPowers_mvdr_dB = 10*log10(beamPowers_mvdr) - max(10*log10(beamPowers_mvdr));

plot(beamAngles, beamPowers_das_dB, 'k-s', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(beamAngles, beamPowers_mvdr_dB, 'm-^', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Beam Direction (degrees)'); ylabel('Relative Power (dB)');
title('Directivity Patterns with Interference (Source: 0°, Interference: 180°)');
legend('DAS', 'MVDR', 'Location', 'best');
grid on; xlim([0, 360]); xticks(0:45:360);

sgtitle('Interference Suppression Performance Analysis', 'FontSize', 14);
drawnow;
saveas(gcf, fullfile(outputDir, 'test2_interference_suppression.png'));
fprintf('Saved: test2_interference_suppression.png\n');

%% Test 3: Covariance Matrix Analysis
fprintf('\n=== Test 3: Covariance Matrix Analysis ===\n');

% Generate signals for covariance analysis
[~, micSignals_clean, ~] = generateTestSignalWithInterference(...
    arrayPos, 0, testFreq, [], [], fs, duration, snr_dB);
[~, micSignals_with_int, ~] = generateTestSignalWithInterference(...
    arrayPos, 0, testFreq, 180, interferenceFreq, fs, duration, snr_dB);

% Compute covariance matrices
R_clean = computeCovarianceMatrix(micSignals_clean, blockSize, overlap);
R_with_interference = computeCovarianceMatrix(micSignals_with_int, blockSize, overlap);

% Analyze covariance matrices at test frequency
freq_bin = round(testFreq * blockSize / fs) + 1;
R_clean_f = squeeze(R_clean(:, :, freq_bin));
R_with_int_f = squeeze(R_with_interference(:, :, freq_bin));

% Plot covariance analysis
figure('Position', [300, 300, 1400, 1000]);
set(gcf, 'Color', 'white');

subplot(2,4,1);
imagesc(abs(R_clean_f));
colorbar; colormap(jet);
title('Covariance Matrix Magnitude (Clean)');
xlabel('Microphone'); ylabel('Microphone');

subplot(2,4,2);
imagesc(abs(R_with_int_f));
colorbar; colormap(jet);
title('Covariance Matrix Magnitude (With Interference)');
xlabel('Microphone'); ylabel('Microphone');

subplot(2,4,3);
imagesc(angle(R_clean_f));
colorbar; colormap(hsv);
title('Covariance Matrix Phase (Clean)');
xlabel('Microphone'); ylabel('Microphone');

subplot(2,4,4);
imagesc(angle(R_with_int_f));
colorbar; colormap(hsv);
title('Covariance Matrix Phase (With Interference)');
xlabel('Microphone'); ylabel('Microphone');

% Eigenvalue analysis
[V_clean, D_clean] = eig(R_clean_f);
[V_int, D_int] = eig(R_with_int_f);

subplot(2,4,5);
eigenvals_clean = diag(D_clean);
eigenvals_int = diag(D_int);
semilogy(1:nMics, sort(eigenvals_clean, 'descend'), 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;



semilogy(1:nMics, sort(eigenvals_int, 'descend'), 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Eigenvalue Index'); ylabel('Eigenvalue Magnitude');
title('Eigenvalue Comparison');
legend('Clean', 'With Interference', 'Location', 'best');
grid on;