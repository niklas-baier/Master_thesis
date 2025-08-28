ver
% Step 1: Define File Paths

filePaths = {
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S03_U01.CH1.wav', ... % Replace with actual file paths
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S03_U01.CH7.wav', ...
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S03_U01.CH4.wav', ...
};

nChannels = length(filePaths); % Number of channels

% Step 2: Read Audio Data from Files
audioData = [];
fs = 0;
for i = 1:nChannels
    [data, currentFs] = audioread(filePaths{i});
    whos data
    if i == 1
        fs = currentFs; % Get sampling rate from the first file
        nSamples = size(data, 1);
        audioData = zeros(nSamples, nChannels); % Preallocate matrix
    else
        % Ensure consistent sampling rate
        if currentFs ~= fs
            error('Sampling rate mismatch between files.');
        end
    end
    % Add the data as a column
    audioData(:, i) = data;
end
% Step 3: Microphone Array Geometry
micSpacing = 0.035; % 5 cm spacing between microphones
array = phased.ULA('NumElements', nChannels, 'ElementSpacing', micSpacing);

% Speed of sound
c = 343; % m/s

% Step 4: Create the Beamformer
% Using Time-Delay Beamformer
targetDirection = [-17; 0]; % Target azimuth and elevation (broadside)
beamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
    'SampleRate', fs, ...
    'PropagationSpeed', c, ...
    'Direction', targetDirection);

% Step 5: Apply Beamforming
% Note: Beamformer expects [nSamples x nChannels] input
beamformedSignal = beamformer(audioData);

% Normalize the beamformed signal


% Step 6: Save the Beamformed Signal
audiowrite('S03_U01.CH1.wav', beamformedSignal, fs);
disp('Beamformed signal saved as "beamformed_output.wav"');

% Inputs
nChannels = 2; % Number of microphone channels
channelFiles = filePaths; % List of .wav files for each channel
outputFile_LCMV = 'beamformed_LCMV.wav';
outputFile_GSC = 'beamformed_GSC.wav';
micSpacing = 0.035; % Microphone spacing in meters
fs = 16000; % Sampling frequency (Hz)
c = 343; % Speed of sound (m/s)

% Step 1: Load multichannel data
disp('Loading multichannel data...');
audioData = []; % Initialize an empty array for audio data
for i = 1:nChannels
    [channelData, fsChannel] = audioread(channelFiles{i});
    if i == 1
        fs = fsChannel; % Set sampling frequency from the first channel
    elseif fs ~= fsChannel
        error('Sampling frequencies of input channels do not match!');
    end
    audioData = [audioData, channelData]; % Concatenate data into multichannel format
end

% Create microphone array
array = phased.ULA('NumElements', nChannels, 'ElementSpacing', micSpacing);

% Step 2: Detect Optimal Direction
disp('Scanning for optimal direction...');
azimuthRange = -90:1:90; % Scan azimuth angles
elevationRange = 0; % Assume flat plane (no elevation changes)
maxPower = -Inf;
optimalDirection = [0; 0];

% Scan the azimuth range


% Define target and null directions
targetDirection = [0; 0]; % Broadside
nullDirection = [30; 0]; % Null at 30 degrees azimuth

% Compute the steering vector for the array
steeringVector = phased.SteeringVector('SensorArray', array, 'PropagationSpeed', c);
targetSteerVec = steeringVector(fc, targetDirection);
nullSteerVec = steeringVector(fc, nullDirection);

% Define the LCMV beamformer
beamformer = phased.LCMVBeamformer( ...
    'Constraint', [targetSteerVec, nullSteerVec], ...
    'DesiredResponse', [1; 0], ... % Target gain of 1, null gain of 0
    'SampleRate', fs);

% Apply the beamformer to the multichannel audio
beamformedSignal = beamformer(audioData);

% Save the beamformed output
audiowrite('LCMV_output.wav', beamformedSignal, fs);
disp('LCMV beamformed signal saved as "beamformed_output.wav"');

% Step 4: GSC Beamforming
disp('Applying GSC Beamformer...');
beamformer_GSC = phased.GSCBeamformer('SensorArray', array, ...
    'PropagationSpeed', c, ...
    'SampleRate', fs, ...
    'Direction', optimalDirection, ...
    'BlockingMatrix', phased.GSCBlockingMatrix('SensorArray', array));
beamformedSignal_GSC = beamformer_GSC(audioData);

% Normalize the beamformed GSC signal
beamformedSignal_GSC = beamformedSignal_GSC / max(abs(beamformedSignal_GSC));

% Save GSC beamformed signal
audiowrite(outputFile_GSC, beamformedSignal_GSC, fs);
disp(['GSC beamformed signal saved as "', outputFile_GSC, '"']);







% Step 7 (Optional): Plot Signals
t = (0:nSamples-1)/fs; % Time vector
figure;
subplot(2, 1, 1);
plot(t, audioData(:, 1)); % First channel of original data
title('Original Signal (First Channel)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
plot(t, beamformedSignal(1:nSamples)); % Beamformed signal
title('Beamformed Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;