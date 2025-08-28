% Improved Beamforming for ASR with Speech Activity Detection
% This code implements beamforming optimized for Automatic Speech Recognition

clc; clear; close all;

%% Step 1: Configuration
% File paths for your microphone channels
filePaths = {
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S03_U01.CH1.wav',
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S03_U01.CH7.wav',
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S03_U01.CH4.wav'
};

% Parameters
micSpacing = 0.035;     % 3.5 cm spacing between microphones
c = 343;                % Speed of sound (m/s)
nChannels = length(filePaths);

% Output files
outputFile = 'beamformed_for_ASR.wav';

%% Step 2: Load Audio Data
fprintf('Loading multichannel audio data...\n');
audioData = [];
fs = [];

for i = 1:nChannels
    [data, currentFs] = audioread(filePaths{i});
    
    if i == 1
        fs = currentFs;
        nSamples = length(data);
        audioData = zeros(nSamples, nChannels);
        fprintf('Loaded %d samples at %d Hz\n', nSamples, fs);
    else
        if currentFs ~= fs
            error('Sampling rate mismatch between files.');
        end
        if length(data) ~= nSamples
            % Handle different lengths by taking minimum
            minLength = min(nSamples, length(data));
            audioData = audioData(1:minLength, :);
            data = data(1:minLength);
            nSamples = minLength;
            fprintf('Warning: Trimmed to %d samples due to length mismatch\n', minLength);
        end
    end
    
    audioData(:, i) = data;
end

%% Step 3: Speech Activity Detection (VAD)
fprintf('Performing Voice Activity Detection...\n');

% Simple energy-based VAD
frameSize = round(0.025 * fs);  % 25ms frames
hopSize = round(0.01 * fs);     % 10ms hop
nFrames = floor((nSamples - frameSize) / hopSize) + 1;

% Compute energy for each frame (using first channel as reference)
energy = zeros(nFrames, 1);
for i = 1:nFrames
    startIdx = (i-1) * hopSize + 1;
    endIdx = min(startIdx + frameSize - 1, nSamples);
    frame = audioData(startIdx:endIdx, 1);
    energy(i) = sum(frame.^2);
end

% Threshold-based VAD
energyThreshold = 0.1 * max(energy);  % Adjust this threshold as needed
speechFrames = energy > energyThreshold;

% Create speech mask for the entire signal
speechMask = false(nSamples, 1);
for i = 1:nFrames
    startIdx = (i-1) * hopSize + 1;
    endIdx = min(startIdx + frameSize - 1, nSamples);
    if speechFrames(i)
        speechMask(startIdx:endIdx) = true;
    end
end

fprintf('Detected %.1f%% speech activity\n', 100 * sum(speechMask) / nSamples);

%% Step 4: Direction Detection Using Speech Segments
fprintf('Detecting optimal direction using speech segments...\n');

% Create microphone array
array = phased.ULA('NumElements', nChannels, 'ElementSpacing', micSpacing);

% Use only speech segments for direction detection
speechIndices = find(speechMask);
if length(speechIndices) < fs  % Need at least 1 second of speech
    warning('Limited speech detected, using all data for direction estimation');
    speechIndices = 1:nSamples;
end

% Sample speech data for direction detection (to speed up processing)
sampleSize = min(length(speechIndices), 5 * fs);  % Use up to 5 seconds
sampleIndices = speechIndices(1:round(length(speechIndices)/sampleSize):end);
sampleIndices = sampleIndices(1:min(end, sampleSize));

speechSample = audioData(sampleIndices, :);

% Scan for optimal direction
azimuthRange = -90:5:90;  % Coarser grid for speed
maxPower = -Inf;
optimalDirection = [0; 0];

% Method 1: Create new beamformer for each direction (recommended)
for az = azimuthRange
    % Create a new beamformer for each direction
    scanBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
        'SampleRate', fs, 'PropagationSpeed', c, 'Direction', [az; 0]);
    
    beamOutput = scanBeamformer(speechSample);
    power = mean(beamOutput.^2);
    
    if power > maxPower
        maxPower = power;
        optimalDirection = [az; 0];
    end
end

% Alternative Method 2: Use release() method (uncomment if preferred)
% scanBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
%     'SampleRate', fs, 'PropagationSpeed', c);
% 
% for az = azimuthRange
%     release(scanBeamformer);  % Release the beamformer
%     scanBeamformer.Direction = [az; 0];  % Now we can change the direction
%     beamOutput = scanBeamformer(speechSample);
%     power = mean(beamOutput.^2);
%     
%     if power > maxPower
%         maxPower = power;
%         optimalDirection = [az; 0];
%     end
% end

fprintf('Optimal direction found: %.1f degrees azimuth\n', optimalDirection(1));

%% Step 5: Apply Adaptive Beamforming
fprintf('Applying adaptive beamforming...\n');

% For ASR, we'll use a combination approach:
% 1. First apply time-domain beamforming for basic directional enhancement
% 2. Then apply adaptive processing for interference suppression

% Method A: Enhanced Time-Delay Beamformer (Primary approach)
fprintf('Applying Time-Delay beamforming with optimal direction...\n');
primaryBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
    'SampleRate', fs, 'PropagationSpeed', c, 'Direction', optimalDirection);

beamformedSignal = primaryBeamformer(audioData);

% Method B: Apply additional adaptive processing using covariance
fprintf('Applying adaptive interference suppression...\n');

% Estimate interference covariance from non-speech segments
noiseIndices = find(~speechMask);
if length(noiseIndices) < fs/10  % Need at least 100ms of noise
    % If no clear noise segments, use beginning and end of recording
    noiseIndices = [1:min(fs/2, nSamples), max(1, nSamples-fs/2+1):nSamples];
end

% For frequency-domain MVDR, we need to work with individual frequency bins
% Let's implement a simplified adaptive approach instead

% Estimate noise characteristics from quiet segments
if ~isempty(noiseIndices)
    noiseSample = audioData(noiseIndices, :);
    
    % Compute spatial covariance matrix
    R_noise = (noiseSample' * noiseSample) / length(noiseIndices);
    
    % Add diagonal loading for numerical stability
    diagLoading = 0.01 * trace(R_noise) / nChannels;
    R_noise = R_noise + diagLoading * eye(nChannels);
    
    % Compute steering vector for target direction
    fc = 1000; % Center frequency for steering vector calculation (1kHz)
    steeringVector = phased.SteeringVector('SensorArray', array, ...
        'PropagationSpeed', c);
    targetSteerVec = steeringVector(fc, optimalDirection);
    
    % Compute MVDR weights
    try
        mvdr_weights = (R_noise \ targetSteerVec) / (targetSteerVec' / R_noise * targetSteerVec);
        
        % Apply MVDR weights to get enhanced signal
        beamformedSignal_MVDR = audioData * conj(mvdr_weights);
        
        % Combine time-delay and MVDR results (weighted average)
        alpha = 0.7; % Weight for time-delay beamformer
        beamformedSignal = alpha * beamformedSignal + (1-alpha) * real(beamformedSignal_MVDR);
        
        fprintf('Successfully applied MVDR enhancement\n');
    catch ME
        fprintf('MVDR processing failed, using time-delay result: %s\n', ME.message);
        % beamformedSignal already contains time-delay result
    end
else
    fprintf('No noise segments detected, using time-delay beamforming only\n');
end

%% Step 6: Post-processing
fprintf('Post-processing beamformed signal...\n');

% Normalize to prevent clipping
maxVal = max(abs(beamformedSignal));
if maxVal > 0
    beamformedSignal = beamformedSignal / maxVal * 0.95;
end

% Optional: Apply mild high-pass filter to remove low-frequency noise
% This is often beneficial for ASR
[b, a] = butter(4, 80/(fs/2), 'high');  % 80 Hz high-pass
beamformedSignal = filtfilt(b, a, beamformedSignal);

%% Step 7: Save Output
audiowrite(outputFile, beamformedSignal, fs);
fprintf('Beamformed signal saved as "%s"\n', outputFile);

%% Step 8: Quality Assessment
fprintf('\nQuality Assessment:\n');

% Signal-to-Noise Ratio estimation
speechPower = mean(beamformedSignal(speechMask).^2);
noisePower = mean(beamformedSignal(~speechMask).^2);
if noisePower > 0
    snr_db = 10 * log10(speechPower / noisePower);
    fprintf('Estimated SNR: %.1f dB\n', snr_db);
end

% Compare with original signal (first channel)
originalSpeechPower = mean(audioData(speechMask, 1).^2);
originalNoisePower = mean(audioData(~speechMask, 1).^2);
if originalNoisePower > 0
    originalSNR = 10 * log10(originalSpeechPower / originalNoisePower);
    fprintf('Original SNR: %.1f dB\n', originalSNR);
    if snr_db > originalSNR
        fprintf('SNR improvement: %.1f dB\n', snr_db - originalSNR);
    end
end

%% Additional Tips for ASR
fprintf('\nTips for ASR optimization:\n');
fprintf('1. The beamformed signal is optimized for speech recognition\n');
fprintf('2. Consider further preprocessing like spectral subtraction if noise persists\n');
fprintf('3. The high-pass filter helps remove low-frequency artifacts\n');
fprintf('4. Monitor the SNR improvement - aim for >3dB improvement\n');
fprintf('5. For very noisy environments, consider multi-stage beamforming\n');