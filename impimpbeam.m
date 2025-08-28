% Improved Beamforming for ASR with Enhanced Speech Activity Detection
% This code implements beamforming optimized for Automatic Speech Recognition

clc; clear; close all;

%% Step 1: Configuration
% File paths for your microphone channels
base_folder = '/media/niklas/SSD2/Dataset/Dipco/audio/eval/';
file_path_1 = 'S08_U05.CH1.wav';

filePaths = {
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S08_U05.CH1.wav',
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S08_U05.CH7.wav',
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S08_U05.CH4.wav'
};


micSpacing = 0.035;     % 3.5 cm spacing between microphones
c = 343;                % Speed of sound (m/s)
nChannels = length(filePaths);

% Output files
target_dir = '/home/niklas/beamforming';
outputFile = fullfile(target_dir,file_path_1);

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

%% Step 3: Enhanced Speech Activity Detection (VAD)
fprintf('Performing Enhanced Voice Activity Detection...\n');

frameSize = round(0.025 * fs);  % 25ms frames
hopSize = round(0.01 * fs);     % 10ms hop
nFrames = floor((nSamples - frameSize) / hopSize) + 1;

% Compute multiple features for each frame (using first channel as reference)
energy = zeros(nFrames, 1);
spectralCentroid = zeros(nFrames, 1);
zeroCrossingRate = zeros(nFrames, 1);

for i = 1:nFrames
    startIdx = (i-1) * hopSize + 1;
    endIdx = min(startIdx + frameSize - 1, nSamples);
    frame = audioData(startIdx:endIdx, 1);
    
    % Energy feature
    energy(i) = sum(frame.^2);
    
    % Spectral centroid (indicates presence of speech formants)
    windowedFrame = frame .* hann(length(frame));
    fftFrame = fft(windowedFrame);
    magnitude = abs(fftFrame(1:floor(length(fftFrame)/2)));
    freqs = (0:length(magnitude)-1) * fs / (2 * length(magnitude));
    
    if sum(magnitude) > 0
        spectralCentroid(i) = sum(freqs' .* magnitude) / sum(magnitude);
    end
    
    % Zero crossing rate (speech has moderate ZCR, noise can be very high or low)
    zeroCrossings = sum(abs(diff(sign(frame))) > 0);
    zeroCrossingRate(i) = zeroCrossings / length(frame);
end

% More adaptive thresholding for noisy environments
% Use percentile-based thresholding instead of max-based
energyThreshold = prctile(energy, 60);  % Use 60th percentile

% Additional constraints for speech-like characteristics
% Speech typically has spectral centroid between 500-3000 Hz
freqMask = (spectralCentroid > 500) & (spectralCentroid < 3000);

% Speech has moderate zero crossing rate
zcrMedian = median(zeroCrossingRate);
zcrStd = std(zeroCrossingRate);
zcrMask = (zeroCrossingRate > zcrMedian - zcrStd) & ...
          (zeroCrossingRate < zcrMedian + 2*zcrStd);

% Combine criteria with more lenient thresholds for noisy environments
speechFrames = (energy > energyThreshold) | ...
               (freqMask & (energy > 0.3 * energyThreshold));

% Apply median filtering to reduce false positives/negatives
speechFrames = medfilt1(double(speechFrames), 5) > 0.5;

% Create speech mask for the entire signal
speechMask = false(nSamples, 1);
for i = 1:nFrames
    startIdx = (i-1) * hopSize + 1;
    endIdx = min(startIdx + frameSize - 1, nSamples);
    if speechFrames(i)
        speechMask(startIdx:endIdx) = true;
    end
end

speechPercentage = 100 * sum(speechMask) / nSamples;
fprintf('Detected %.1f%% speech activity\n', speechPercentage);

% If still very low, fall back to more lenient threshold
if speechPercentage < 5
    fprintf('Speech activity very low, applying more lenient thresholding...\n');
    energyThreshold = prctile(energy, 40);  % Even more lenient
    speechFrames = energy > energyThreshold;
    
    % Recreate speech mask
    speechMask = false(nSamples, 1);
    for i = 1:nFrames
        startIdx = (i-1) * hopSize + 1;
        endIdx = min(startIdx + frameSize - 1, nSamples);
        if speechFrames(i)
            speechMask(startIdx:endIdx) = true;
        end
    end
    
    speechPercentage = 100 * sum(speechMask) / nSamples;
    fprintf('Revised speech activity: %.1f%%\n', speechPercentage);
end

%% Step 4: Enhanced Direction Detection Using Speech Segments
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
stepSize = max(1, floor(length(speechIndices) / sampleSize));
sampleIndices = speechIndices(1:stepSize:end);
sampleIndices = sampleIndices(1:min(end, sampleSize));

speechSample = audioData(sampleIndices, :);

% Coarse scan for optimal direction (5 degree steps like Python)
azimuthRange = -90:5:90;
maxPower = -Inf;
optimalDirection = [0; 0];

fprintf('Scanning directions: ');
for az = azimuthRange
    scanBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
        'SampleRate', fs, 'PropagationSpeed', c, 'Direction', [az; 0]);
    
    beamOutput = scanBeamformer(speechSample);
    power = mean(beamOutput.^2);
    
    if power > maxPower
        maxPower = power;
        optimalDirection = [az; 0];
    end
    
    fprintf('%d° ', az);
end
fprintf('\n');

% Fine-tune around optimal direction with 1° resolution
fineRange = (optimalDirection(1)-4):1:(optimalDirection(1)+4);
fineRange = fineRange(fineRange >= -90 & fineRange <= 90);
fprintf('Fine-tuning around %.0f° with 1° resolution...\n', optimalDirection(1));

for az = fineRange
    scanBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
        'SampleRate', fs, 'PropagationSpeed', c, 'Direction', [az; 0]);
    
    beamOutput = scanBeamformer(speechSample);
    power = mean(beamOutput.^2);
    
    if power > maxPower
        maxPower = power;
        optimalDirection = [az; 0];
    end
end

fprintf('Optimal direction found: %.1f degrees azimuth\n', optimalDirection(1));

%% Step 5: Apply Multi-Stage Adaptive Beamforming
fprintf('Applying multi-stage adaptive beamforming...\n');

% Stage 1: Time-Delay Beamforming with optimal direction
fprintf('Stage 1: Time-Delay beamforming with optimal direction...\n');
primaryBeamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
    'SampleRate', fs, 'PropagationSpeed', c, 'Direction', optimalDirection);

beamformedSignal = primaryBeamformer(audioData);

% Stage 2: Spectral Subtraction for additional noise reduction
fprintf('Stage 2: Applying spectral subtraction...\n');
beamformedSignal = applySpectralSubtraction(beamformedSignal, fs, speechMask);

% Stage 3: Adaptive MVDR processing
fprintf('Stage 3: Applying adaptive interference suppression...\n');

% Estimate interference covariance from non-speech segments
noiseIndices = find(~speechMask);
if length(noiseIndices) < fs/10  % Need at least 100ms of noise
    % Use beginning and end of recording plus detected noise segments
    beginIndices = 1:min(fs/2, nSamples);
    endIndices = max(1, nSamples-fs/2+1):nSamples;
    noiseIndices = [beginIndices, endIndices];
end

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
        
        % Adaptive combination based on SNR estimate
        localSNR = estimateLocalSNR(beamformedSignal, speechMask, fs);
        if localSNR < 5  % Low SNR case
            alpha = 0.5;  % Equal weighting
        else
            alpha = 0.7;  % Favor time-delay beamformer
        end
        
        beamformedSignal = alpha * beamformedSignal + (1-alpha) * real(beamformedSignal_MVDR);
        
        fprintf('Successfully applied MVDR enhancement with alpha=%.1f\n', alpha);
    catch ME
        fprintf('MVDR processing failed, using time-delay result: %s\n', ME.message);
    end
else
    fprintf('No noise segments detected, using time-delay beamforming only\n');
end

%% Step 6: Enhanced Post-processing
fprintf('Enhanced post-processing...\n');

% Normalize to prevent clipping
maxVal = max(abs(beamformedSignal));
if maxVal > 0
    beamformedSignal = beamformedSignal / maxVal * 0.95;
end

% Multi-stage filtering
% Stage 1: High-pass filter to remove low-frequency noise
[b1, a1] = butter(4, 80/(fs/2), 'high');  % 80 Hz high-pass
beamformedSignal = filtfilt(b1, a1, beamformedSignal);

% Stage 2: Notch filter for common interference frequencies (optional)
% Uncomment if you have specific interference frequencies
% [b2, a2] = butter(4, [49 51]/(fs/2), 'stop');  % 50 Hz notch
% beamformedSignal = filtfilt(b2, a2, beamformedSignal);

% Stage 3: Speech enhancement filter (emphasis on speech frequencies)
[b3, a3] = butter(2, [300 3400]/(fs/2), 'bandpass');  % Speech band
speechEnhanced = filtfilt(b3, a3, beamformedSignal);

% Combine original and speech-enhanced versions
beamformedSignal = 0.8 * beamformedSignal + 0.2 * speechEnhanced;

% Final normalization
maxVal = max(abs(beamformedSignal));
if maxVal > 0
    beamformedSignal = beamformedSignal / maxVal * 0.95;
end

%% Step 7: Save Output
audiowrite(outputFile, beamformedSignal, fs);
fprintf('Enhanced beamformed signal saved as "%s"\n', outputFile);

%% Step 8: Comprehensive Quality Assessment
fprintf('\nComprehensive Quality Assessment:\n');

% Signal-to-Noise Ratio estimation
speechPower = mean(beamformedSignal(speechMask).^2);
noisePower = mean(beamformedSignal(~speechMask).^2);
if noisePower > 0
    snr_db = 10 * log10(speechPower / noisePower);
    fprintf('Final SNR: %.1f dB\n', snr_db);
else
    snr_db = Inf;
    fprintf('Final SNR: Inf dB\n');
end

% Compare with original signal (first channel)
originalSpeechPower = mean(audioData(speechMask, 1).^2);
originalNoisePower = mean(audioData(~speechMask, 1).^2);
if originalNoisePower > 0
    originalSNR = 10 * log10(originalSpeechPower / originalNoisePower);
    fprintf('Original SNR: %.1f dB\n', originalSNR);
    if isfinite(snr_db) && snr_db > originalSNR
        fprintf('SNR improvement: %.1f dB\n', snr_db - originalSNR);
    end
end

% Additional quality metrics
fprintf('Additional Metrics:\n');
fprintf('- Speech segments detected: %.1f%%\n', speechPercentage);
fprintf('- Optimal beam direction: %.1f°\n', optimalDirection(1));
fprintf('- Processing stages applied: Time-Delay + Spectral Subtraction + MVDR\n');

%% Helper Functions

function enhancedSignal = applySpectralSubtraction(signal, fs, speechMask)
    % Simple spectral subtraction implementation
    
    % Parameters
    frameSize = round(0.032 * fs);  % 32ms frames
    hopSize = round(0.016 * fs);    % 16ms hop (50% overlap)
    alpha = 2.0;  % Over-subtraction factor
    beta = 0.1;   % Spectral floor factor
    
    % Get noise spectrum from non-speech segments
    noiseIndices = find(~speechMask);
    if isempty(noiseIndices)
        enhancedSignal = signal;
        return;
    end
    
    % Estimate noise spectrum
    noiseSegment = signal(noiseIndices(1:min(end, frameSize*10)));  % Use up to 10 frames
    noiseSpectrum = abs(fft(noiseSegment .* hann(length(noiseSegment))));
    noiseSpectrum = noiseSpectrum(1:floor(length(noiseSpectrum)/2)+1);
    
    % Process signal in overlapping frames
    nFrames = floor((length(signal) - frameSize) / hopSize) + 1;
    enhancedSignal = zeros(size(signal));
    window = hann(frameSize);
    
    for i = 1:nFrames
        startIdx = (i-1) * hopSize + 1;
        endIdx = startIdx + frameSize - 1;
        
        if endIdx > length(signal)
            break;
        end
        
        % Extract and window frame
        frame = signal(startIdx:endIdx) .* window;
        
        % FFT
        frameSpectrum = fft(frame);
        frameMagnitude = abs(frameSpectrum);
        framePhase = angle(frameSpectrum);
        
        % Spectral subtraction
        enhancedMagnitude = frameMagnitude(1:floor(length(frameMagnitude)/2)+1);
        noiseEst = noiseSpectrum(1:length(enhancedMagnitude));
        
        % Apply subtraction with over-subtraction and spectral floor
        enhancedMagnitude = enhancedMagnitude - alpha * noiseEst;
        enhancedMagnitude = max(enhancedMagnitude, beta * frameMagnitude(1:length(enhancedMagnitude)));
        
        % Reconstruct full spectrum
        enhancedSpectrum = [enhancedMagnitude; flipud(enhancedMagnitude(2:end-1))];
        enhancedSpectrum = enhancedSpectrum .* exp(1j * framePhase);
        
        % IFFT and overlap-add
        enhancedFrame = real(ifft(enhancedSpectrum)) .* window;
        enhancedSignal(startIdx:endIdx) = enhancedSignal(startIdx:endIdx) + enhancedFrame;
    end
end

function snr = estimateLocalSNR(signal, speechMask, fs)
    % Estimate local SNR for adaptive processing
    
    % Use sliding window approach
    windowSize = fs; % 1 second windows
    nWindows = floor(length(signal) / windowSize);
    
    snrValues = [];
    for i = 1:nWindows
        startIdx = (i-1) * windowSize + 1;
        endIdx = startIdx + windowSize - 1;
        
        windowMask = speechMask(startIdx:endIdx);
        windowSignal = signal(startIdx:endIdx);
        
        if sum(windowMask) > 0.3 * windowSize  % At least 30% speech
            speechPower = mean(windowSignal(windowMask).^2);
            noisePower = mean(windowSignal(~windowMask).^2);
            
            if noisePower > 0
                snrValues(end+1) = 10 * log10(speechPower / noisePower);
            end
        end
    end
    
    if isempty(snrValues)
        snr = 0;
    else
        snr = median(snrValues);
    end
end