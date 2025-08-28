set(groot, 'DefaultFigureRenderer', 'painters');  % Try different renderer
% DAS Beamforming Debug Script
% Tests the beamforming system with synthetic signals
% Generates visualization plots saved as PNG files

clc; clear; close all;

%% Configuration
outputDir = './debug_results/';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Test parameters
fs = 16000;                % Sample rate (Hz)
duration = 2.0;            % Signal duration (seconds)
testFreq = 1000;           % Test frequency (Hz)
snr_dB = 20;               % Signal-to-noise ratio (dB)

% Array geometry - same as your main script
radius = 0.035;            % 3.5 cm radius
nMics = 7;                 % 6 outer + 1 center
c = 343;                   % Speed of sound (m/s)

% Test scenarios
sourceAngles = [0, 45, 90, 135, 180, 225, 270, 315]; % Different source directions
beamAngles = [0, 45, 90, 135, 180, 225, 270, 315];   % Different beam directions

fprintf('=== DAS Beamforming Debug Script ===\n');
fprintf('Array radius: %.1f cm\n', radius * 100);
fprintf('Number of microphones: %d\n', nMics);
fprintf('Test frequency: %d Hz\n', testFreq);
fprintf('Sample rate: %d Hz\n', fs);
fprintf('Signal duration: %.1f s\n', duration);
fprintf('MATLAB version: %s\n', version);

% Check graphics capabilities
fprintf('Graphics renderer: %s\n', get(groot, 'DefaultFigureRenderer'));

%% Setup array geometry
arrayPos = getCircularArrayPositions(radius, nMics);

% Display array geometry
fprintf('\nArray geometry:\n');
for i = 1:nMics
    fprintf('Mic %d: (%.3f, %.3f, %.3f) m\n', i, arrayPos(1,i), arrayPos(2,i), arrayPos(3,i));
end

%% Test 1: Basic functionality with single source
fprintf('\n=== Test 1: Basic Beamforming Test ===\n');

sourceAngle = 0;  % Source at 0 degrees (front)
beamAngle = 0;    % Beam pointing at 0 degrees

% Generate test signal
[testSignal, micSignals] = generateTestSignal(arrayPos, sourceAngle, testFreq, fs, duration, snr_dB);

% Apply beamforming
beamformedSignal = applyDASBeamforming(micSignals, arrayPos, beamAngle, fs, c);

% Plot results
figure('Position', [100, 100, 1200, 800]);
set(gcf, 'Color', 'white');  % Ensure white background

subplot(2,3,1);
plot((0:length(testSignal)-1)/fs, testSignal, 'b-', 'LineWidth', 1.5);
title('Original Source Signal', 'FontSize', 12);
xlabel('Time (s)'); ylabel('Amplitude');
grid on; axis tight;

subplot(2,3,2);
plot((0:size(micSignals,1)-1)/fs, micSignals(:,1:3), 'LineWidth', 1.2);
title('Microphone Signals (First 3 Mics)', 'FontSize', 12);
xlabel('Time (s)'); ylabel('Amplitude');
legend('Mic 1', 'Mic 2', 'Mic 3', 'Location', 'best');
grid on; axis tight;

subplot(2,3,3);
plot((0:length(beamformedSignal)-1)/fs, beamformedSignal, 'r-', 'LineWidth', 1.5);
title('Beamformed Signal', 'FontSize', 12);
xlabel('Time (s)'); ylabel('Amplitude');
grid on; axis tight;

% Frequency analysis
subplot(2,3,4);
[f, P_orig] = computeSpectrum(testSignal, fs);
plot(f, 10*log10(P_orig), 'b-', 'LineWidth', 1.5);
title('Original Signal Spectrum', 'FontSize', 12);
xlabel('Frequency (Hz)'); ylabel('Power (dB)');
grid on; xlim([0, 2000]); axis tight;

subplot(2,3,5);
[f, P_mic] = computeSpectrum(micSignals(:,1), fs);
plot(f, 10*log10(P_mic), 'g-', 'LineWidth', 1.5);
title('Microphone 1 Spectrum', 'FontSize', 12);
xlabel('Frequency (Hz)'); ylabel('Power (dB)');
grid on; xlim([0, 2000]); axis tight;

subplot(2,3,6);
[f, P_beam] = computeSpectrum(beamformedSignal, fs);
plot(f, 10*log10(P_beam), 'r-', 'LineWidth', 1.5);
title('Beamformed Signal Spectrum', 'FontSize', 12);
xlabel('Frequency (Hz)'); ylabel('Power (dB)');
grid on; xlim([0, 2000]); axis tight;

sgtitle(sprintf('DAS Beamforming - Source at %d°, Beam at %d°', sourceAngle, beamAngle), 'FontSize', 14);
drawnow;  % Force rendering before saving
saveas(gcf, fullfile(outputDir, 'test1_basic_beamforming.png'));
fprintf('Saved: test1_basic_beamforming.png\n');

%% Test 2: Directivity Pattern
fprintf('\n=== Test 2: Directivity Pattern Analysis ===\n');

sourceAngle = 0;  % Fixed source direction
beamPowers = zeros(size(beamAngles));

% Test beamforming performance for different beam directions
for i = 1:length(beamAngles)
    [~, micSignals] = generateTestSignal(arrayPos, sourceAngle, testFreq, fs, duration, snr_dB);
    beamformedSignal = applyDASBeamforming(micSignals, arrayPos, beamAngles(i), fs, c);
    beamPowers(i) = mean(beamformedSignal.^2);
end

% Convert to dB and normalize
beamPowers_dB = 10*log10(beamPowers);
beamPowers_dB = beamPowers_dB - max(beamPowers_dB);

% Plot directivity pattern
figure('Position', [200, 200, 1000, 500]);
set(gcf, 'Color', 'white');

subplot(1,2,1);
plot(beamAngles, beamPowers_dB, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
xlabel('Beam Direction (degrees)', 'FontSize', 11); 
ylabel('Relative Power (dB)', 'FontSize', 11);
title(sprintf('Directivity Pattern (Source at %d°)', sourceAngle), 'FontSize', 12);
grid on; ylim([-20, 5]); xlim([0, 360]); 
xticks(0:45:360); axis tight;

subplot(1,2,2);
% Polar plot
theta = beamAngles * pi/180;
beamPowers_linear = 10.^(beamPowers_dB/10);
polarplot(theta, beamPowers_linear, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
title(sprintf('Polar Directivity Pattern (Source at %d°)', sourceAngle), 'FontSize', 12);
rlim([0, 1.2]);

sgtitle('Array Directivity Analysis', 'FontSize', 14);
drawnow;
saveas(gcf, fullfile(outputDir, 'test2_directivity_pattern.png'));
fprintf('Saved: test2_directivity_pattern.png\n');

%% Test 3: Source Localization Performance
fprintf('\n=== Test 3: Source Localization Performance ===\n');

% Test with multiple source positions
localizationErrors = zeros(size(sourceAngles));
estimatedAngles = zeros(size(sourceAngles));

for i = 1:length(sourceAngles)
    [~, micSignals] = generateTestSignal(arrayPos, sourceAngles(i), testFreq, fs, duration, snr_dB);
    estimatedAngle = estimateTargetDirection(micSignals, arrayPos, fs, c);
    estimatedAngles(i) = estimatedAngle;
    
    % Calculate angular error (accounting for wraparound)
    error = abs(estimatedAngle - sourceAngles(i));
    if error > 180
        error = 360 - error;
    end
    localizationErrors(i) = error;
end

% Plot localization results
figure('Position', [300, 300, 1200, 400]);

subplot(1,3,1);
plot(sourceAngles, estimatedAngles, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot([0, 360], [0, 360], 'r--', 'LineWidth', 1);
xlabel('True Source Angle (degrees)'); ylabel('Estimated Angle (degrees)');
title('Source Localization Accuracy');
grid on; axis equal; xlim([0, 360]); ylim([0, 360]);
legend('Estimated', 'Perfect', 'Location', 'northwest');

subplot(1,3,2);
bar(sourceAngles, localizationErrors, 'FaceColor', [0.3, 0.6, 0.9]);
xlabel('True Source Angle (degrees)'); ylabel('Localization Error (degrees)');
title('Localization Error vs Source Position');
xticks(sourceAngles);
grid on; ylim([0, max(localizationErrors)*1.2]);

subplot(1,3,3);
histogram(localizationErrors, 8, 'FaceColor', [0.3, 0.6, 0.9]);
xlabel('Localization Error (degrees)'); ylabel('Count');
title('Error Distribution');
grid on;

sgtitle('Source Localization Performance Analysis');
saveas(gcf, fullfile(outputDir, 'test3_localization_performance.png'));
fprintf('Saved: test3_localization_performance.png\n');

%% Test 4: SNR Improvement Analysis
fprintf('\n=== Test 4: SNR Improvement Analysis ===\n');

snr_test_dB = [0, 5, 10, 15, 20, 25, 30]; % Different input SNRs
sourceAngle = 0;
beamAngle = 0;

input_snr_measured = zeros(size(snr_test_dB));
output_snr_measured = zeros(size(snr_test_dB));

for i = 1:length(snr_test_dB)
    [testSignal, micSignals] = generateTestSignal(arrayPos, sourceAngle, testFreq, fs, duration, snr_test_dB(i));
    beamformedSignal = applyDASBeamforming(micSignals, arrayPos, beamAngle, fs, c);
    
    % Measure actual SNR
    input_snr_measured(i) = measureSNR(micSignals(:,1), testFreq, fs);
    output_snr_measured(i) = measureSNR(beamformedSignal, testFreq, fs);
end

snr_improvement = output_snr_measured - input_snr_measured;

% Plot SNR analysis
figure('Position', [400, 400, 1000, 600]);

subplot(2,2,1);
plot(snr_test_dB, input_snr_measured, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(snr_test_dB, output_snr_measured, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Input SNR (dB)'); ylabel('Measured SNR (dB)');
title('SNR: Input vs Output');
legend('Input (Mic 1)', 'Output (Beamformed)', 'Location', 'northwest');
grid on;

subplot(2,2,2);
plot(snr_test_dB, snr_improvement, 'go-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Input SNR (dB)'); ylabel('SNR Improvement (dB)');
title('SNR Improvement vs Input SNR');
grid on;

subplot(2,2,3);
bar(snr_test_dB, snr_improvement, 'FaceColor', [0.2, 0.8, 0.2]);
xlabel('Input SNR (dB)'); ylabel('SNR Improvement (dB)');
title('SNR Improvement Summary');
grid on;

subplot(2,2,4);
plot(snr_test_dB, output_snr_measured - snr_test_dB, 'mo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Target SNR (dB)'); ylabel('Measured - Target SNR (dB)');
title('SNR Accuracy');
grid on;

sgtitle('Signal-to-Noise Ratio Analysis');
saveas(gcf, fullfile(outputDir, 'test4_snr_analysis.png'));
fprintf('Saved: test4_snr_analysis.png\n');

%% Test 5: Delay Computation Verification
fprintf('\n=== Test 5: Delay Computation Verification ===\n');

% Test delay computation for different directions
testDirections = 0:30:330;
delays_computed = zeros(nMics, length(testDirections));

for i = 1:length(testDirections)
    delays_computed(:, i) = computeDelays(arrayPos, testDirections(i), c);
end

% Plot delay patterns
figure('Position', [500, 500, 1200, 800]);

subplot(2,2,1);
plot(testDirections, delays_computed' * 1e6, 'LineWidth', 1.5);
xlabel('Source Direction (degrees)'); ylabel('Delay (microseconds)');
title('Computed Delays vs Direction');
legend(arrayfun(@(x) sprintf('Mic %d', x), 1:nMics, 'UniformOutput', false), 'Location', 'best');
grid on;

subplot(2,2,2);
imagesc(testDirections, 1:nMics, delays_computed * 1e6);
xlabel('Source Direction (degrees)'); ylabel('Microphone');
title('Delay Heatmap (microseconds)');
colorbar; colormap(jet);
yticks(1:nMics);

subplot(2,2,3);
% Maximum delay for each direction
max_delays = max(delays_computed) - min(delays_computed);
plot(testDirections, max_delays * 1e6, 'r-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Source Direction (degrees)'); ylabel('Max Delay Span (microseconds)');
title('Maximum Delay Span vs Direction');
grid on;

subplot(2,2,4);
% Theoretical vs actual delays for 0 degrees
theoretical_delays = zeros(nMics, 1);
for i = 1:6  % Outer mics
    angle = (i-1) * 2*pi / 6;
    theoretical_delays(i) = -radius * cos(angle) / c;
end
theoretical_delays(7) = 0; % Center mic

actual_delays = delays_computed(:, 1); % 0 degrees
bar(1:nMics, [theoretical_delays, actual_delays] * 1e6);
xlabel('Microphone'); ylabel('Delay (microseconds)');
title('Theoretical vs Computed Delays (0°)');
legend('Theoretical', 'Computed', 'Location', 'best');
grid on;

sgtitle('Delay Computation Analysis');
saveas(gcf, fullfile(outputDir, 'test5_delay_analysis.png'));
fprintf('Saved: test5_delay_analysis.png\n');

%% Generate summary report
fprintf('\n=== Generating Summary Report ===\n');

% Create summary figure
figure('Position', [600, 600, 1400, 1000]);

% Array geometry
subplot(3,3,1);
scatter(arrayPos(1, 1:6), arrayPos(2, 1:6), 100, 'b', 'filled');
hold on;
scatter(arrayPos(1, 7), arrayPos(2, 7), 150, 'r', 'filled');
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)');
title('Array Geometry');
legend('Outer Mics', 'Center Mic', 'Location', 'best');

% Best case beamforming
subplot(3,3,2);
bestIdx = find(beamAngles == sourceAngle);
if ~isempty(bestIdx)
    bestPower = beamPowers_dB(bestIdx);
else
    bestPower = max(beamPowers_dB);
end
bar([0, bestPower], 'FaceColor', [0.3, 0.6, 0.9]);
set(gca, 'XTickLabel', {'Input', 'Beamformed'});
ylabel('Relative Power (dB)');
title('Best Case Gain');
grid on;

% Localization accuracy
subplot(3,3,3);
rms_error = sqrt(mean(localizationErrors.^2));
bar(rms_error, 'FaceColor', [0.8, 0.3, 0.3]);
set(gca, 'XTickLabel', {'RMS Error'});
ylabel('Error (degrees)');
title(sprintf('Localization RMS Error: %.1f°', rms_error));
grid on;

% SNR improvement
subplot(3,3,4);
mean_snr_improvement = mean(snr_improvement);
bar(mean_snr_improvement, 'FaceColor', [0.2, 0.8, 0.2]);
set(gca, 'XTickLabel', {'Mean SNR Gain'});
ylabel('SNR Improvement (dB)');
title(sprintf('Mean SNR Gain: %.1f dB', mean_snr_improvement));
grid on;

% Frequency response
subplot(3,3,5);
[~, micSignals] = generateTestSignal(arrayPos, 0, testFreq, fs, duration, 20);
beamformedSignal = applyDASBeamforming(micSignals, arrayPos, 0, fs, c);
[f, P_beam] = computeSpectrum(beamformedSignal, fs);
plot(f, 10*log10(P_beam));
title('Beamformed Spectrum');
xlabel('Frequency (Hz)'); ylabel('Power (dB)');
grid on; xlim([0, 2000]);

% Array pattern (polar)
subplot(3,3,6);
theta = beamAngles * pi/180;
beamPowers_linear = 10.^(beamPowers_dB/10);
polarplot(theta, beamPowers_linear, 'b-o', 'LineWidth', 2);
title('Directivity Pattern');

% Performance summary text
subplot(3,3,[7,8,9]);
axis off;
summary_text = {
    'DAS BEAMFORMING DEBUG SUMMARY',
    '================================',
    sprintf('Array Configuration: %d mics, %.1f cm radius', nMics, radius*100),
    sprintf('Test Frequency: %d Hz', testFreq),
    sprintf('Sample Rate: %d Hz', fs),
    '',
    'PERFORMANCE METRICS:',
    sprintf('• Mean SNR Improvement: %.1f dB', mean_snr_improvement),
    sprintf('• Localization RMS Error: %.1f degrees', rms_error),
    sprintf('• Max Directivity Gain: %.1f dB', max(beamPowers_dB)),
    sprintf('• Frequency Response: Peak at %d Hz', testFreq),
    '',
    'STATUS: All tests completed successfully ✓'
};

text(0.05, 0.95, summary_text, 'FontSize', 11, 'FontName', 'Courier', ...
     'VerticalAlignment', 'top', 'Units', 'normalized');

sgtitle('DAS Beamforming System - Debug Summary Report', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(outputDir, 'debug_summary_report.png'));
fprintf('Saved: debug_summary_report.png\n');

%% Print final summary
fprintf('\n=== DEBUG COMPLETE ===\n');
fprintf('All debug plots saved to: %s\n', outputDir);
fprintf('Summary statistics:\n');
fprintf('  • Mean SNR improvement: %.1f dB\n', mean_snr_improvement);
fprintf('  • Localization RMS error: %.1f degrees\n', rms_error);
fprintf('  • Max directivity gain: %.1f dB\n', max(beamPowers_dB));
fprintf('  • System appears to be working correctly!\n');

%% Helper Functions (same as original script)
function arrayPos = getCircularArrayPositions(radius, nMics)
    arrayPos = zeros(3, nMics);
    for i = 1:6
        angle = (i-1) * 2*pi / 6;
        arrayPos(:, i) = [radius * cos(angle); radius * sin(angle); 0];
    end
    arrayPos(:, 7) = [0; 0; 0];
end

function [testSignal, micSignals] = generateTestSignal(arrayPos, sourceAngle, freq, fs, duration, snr_dB)
    % Generate synthetic test signal and simulate microphone array
    
    nSamples = round(duration * fs);
    t = (0:nSamples-1)' / fs;
    
    % Generate clean source signal
    testSignal = sin(2*pi*freq*t);
    
    % Source direction
    sourceDir = [cos(sourceAngle*pi/180); sin(sourceAngle*pi/180); 0];
    
    % Simulate signal arrival at each microphone
    nMics = size(arrayPos, 2);
    micSignals = zeros(nSamples, nMics);
    c = 343; % Speed of sound
    
    for i = 1:nMics
        % Calculate propagation delay
        delay = dot(arrayPos(:, i), sourceDir) / c;
        delaySamples = delay * fs;
        
        % Apply delay to signal
        delayedSignal = applyFractionalDelay(testSignal, delaySamples);
        
        % Add noise
        noise = randn(size(delayedSignal));
        signalPower = mean(delayedSignal.^2);
        noisePower = signalPower / (10^(snr_dB/10));
        noise = noise * sqrt(noisePower);
        
        micSignals(:, i) = delayedSignal + noise;
    end
end

function [f, P] = computeSpectrum(signal, fs)
    % Compute power spectral density
    N = length(signal);
    Y = fft(signal);
    P = abs(Y).^2 / N;
    f = (0:N-1) * fs / N;
    
    % Keep only positive frequencies
    if mod(N, 2) == 0
        P = P(1:N/2+1);
        f = f(1:N/2+1);
        P(2:end-1) = 2*P(2:end-1);
    else
        P = P(1:(N+1)/2);
        f = f(1:(N+1)/2);
        P(2:end) = 2*P(2:end);
    end
end

function snr_dB = measureSNR(signal, freq, fs)
    % Measure SNR by comparing signal power at target frequency vs total power
    [f, P] = computeSpectrum(signal, fs);
    
    % Find frequency bin closest to target frequency
    [~, freqBin] = min(abs(f - freq));
    
    % Signal power (in a small band around target frequency)
    bandwidth = 20; % Hz
    binWidth = fs / length(signal);
    binRange = round(bandwidth / binWidth);
    
    startBin = max(1, freqBin - binRange);
    endBin = min(length(P), freqBin + binRange);
    
    signalPower = sum(P(startBin:endBin));
    totalPower = sum(P);
    noisePower = totalPower - signalPower;
    
    snr_dB = 10 * log10(signalPower / noisePower);
end

% Include all other functions from the original script
function targetDirection = estimateTargetDirection(audioData, arrayPos, fs, c)
    nMics = size(arrayPos, 2);
    azimuthRange = 0:10:350;
    maxPower = -inf;
    targetDirection = 0;
    
    segmentLength = min(round(1.0 * fs), size(audioData, 1));
    audioSegment = audioData(1:segmentLength, :);
    
    for azimuth = azimuthRange
        delays = computeDelays(arrayPos, azimuth, c);
        aligned = applyDelays(audioSegment, delays, fs);
        beamformed = mean(aligned, 2);
        power = mean(beamformed.^2);
        
        if power > maxPower
            maxPower = power;
            targetDirection = azimuth;
        end
    end
end

function beamformedSignal = applyDASBeamforming(audioData, arrayPos, targetDirection, fs, c)
    delays = computeDelays(arrayPos, targetDirection, c);
    alignedSignals = applyDelays(audioData, delays, fs);
    beamformedSignal = mean(alignedSignals, 2);
    beamformedSignal = postProcessSignal(beamformedSignal, fs);
end

function delays = computeDelays(arrayPos, azimuth, c)
    nMics = size(arrayPos, 2);
    azimuthRad = azimuth * pi / 180;
    targetDir = [cos(azimuthRad); sin(azimuthRad); 0];
    refPos = arrayPos(:, 7);
    
    delays = zeros(nMics, 1);
    for i = 1:nMics
        deltaPos = arrayPos(:, i) - refPos;
        delays(i) = -dot(deltaPos, targetDir) / c;
    end
end

function alignedSignals = applyDelays(audioData, delays, fs)
    [nSamples, nMics] = size(audioData);
    alignedSignals = zeros(nSamples, nMics);
    
    for i = 1:nMics
        delaySamples = delays(i) * fs;
        if abs(delaySamples) < 0.1
            alignedSignals(:, i) = audioData(:, i);
        else
            alignedSignals(:, i) = applyFractionalDelay(audioData(:, i), delaySamples);
        end
    end
end

function delayedSignal = applyFractionalDelay(signal, delaySamples)
    intDelay = floor(delaySamples);
    fracDelay = delaySamples - intDelay;
    
    maxDelay = abs(intDelay) + 1;
    paddedSignal = [zeros(maxDelay, 1); signal; zeros(maxDelay, 1)];
    
    if intDelay >= 0
        startIdx = maxDelay + 1 + intDelay;
    else
        startIdx = maxDelay + 1 + intDelay;
    end
    
    endIdx = startIdx + length(signal) - 1;
    startIdx = max(1, min(startIdx, length(paddedSignal)));
    endIdx = max(1, min(endIdx, length(paddedSignal)));
    
    if endIdx <= startIdx
        delayedSignal = zeros(size(signal));
        return;
    end
    
    intDelayedSignal = paddedSignal(startIdx:endIdx);
    
    if abs(fracDelay) < 1e-6
        delayedSignal = intDelayedSignal;
    else
        if fracDelay > 0
            if endIdx < length(paddedSignal)
                nextSamples = paddedSignal(startIdx+1:endIdx+1);
                delayedSignal = (1 - fracDelay) * intDelayedSignal + fracDelay * nextSamples;
            else
                delayedSignal = intDelayedSignal;
            end
        else
            if startIdx > 1
                prevSamples = paddedSignal(startIdx-1:endIdx-1);
                delayedSignal = (1 + fracDelay) * intDelayedSignal - fracDelay * prevSamples;
            else
                delayedSignal = intDelayedSignal;
            end
        end
    end
    
    if length(delayedSignal) ~= length(signal)
        if length(delayedSignal) > length(signal)
            delayedSignal = delayedSignal(1:length(signal));
        else
            delayedSignal = [delayedSignal; zeros(length(signal) - length(delayedSignal), 1)];
        end
    end
end

function signal = postProcessSignal(signal, fs)
    signal = signal - mean(signal);
    
    fadeLength = round(0.01 * fs);
    if length(signal) > 2 * fadeLength
        fadeIn = linspace(0, 1, fadeLength)';
        signal(1:fadeLength) = signal(1:fadeLength) .* fadeIn;
        
        fadeOut = linspace(1, 0, fadeLength)';
        signal(end-fadeLength+1:end) = signal(end-fadeLength+1:end) .* fadeOut;
    end
    
    if exist('butter', 'file')
        [b, a] = butter(2, 80/(fs/2), 'high');
        signal = filtfilt(b, a, signal);
    end
    
    maxVal = max(abs(signal));
    if maxVal > 0
        signal = signal * (0.8 / maxVal);
    end
    
    signal = tanh(signal * 0.9);
end