% Define the base folder path
folderPath = '/media/niklas/SSD2/Dataset/Dipco/audio/dev';
outputFolder = '/media/niklas/SSD2/Dataset/Dipco/audio/output'; % Folder to save beamformed files

% Define the regular expression pattern for the file name
pattern = '^S(\d{2})_U(\d{2})\.CH(\d)\.wav$';

% Get a list of all files in the folder
fileList = dir(fullfile(folderPath, '*.wav'));

% Initialize cell arrays for the three groups
group1 = {};  % Group [1,7,4]
group2 = {};  % Group [2,7,5]
group3 = {};  % Group [6,7,3]

% Loop through each file and group them based on channels
for i = 1:length(fileList)
    tokens = regexp(fileList(i).name, pattern, 'tokens');
    if ~isempty(tokens)
        channel = str2double(tokens{1}{3});
        if ismember(channel, [1, 7, 4])
            group1{end+1} = fullfile(folderPath, fileList(i).name); % Full path
        elseif ismember(channel, [2, 7, 5])
            group2{end+1} = fullfile(folderPath, fileList(i).name); % Full path
        elseif ismember(channel, [6, 7, 3])
            group3{end+1} = fullfile(folderPath, fileList(i).name); % Full path
        end
    end
end

% Combine groups into a list for processing
allGroups = {group1, group2, group3};
groupNames = {'Group1', 'Group2', 'Group3'};

% Beamforming Parameters
micSpacing = 0.035; % Microphone spacing in meters
c = 343; % Speed of sound in m/s

% Process each group
for g = 1:length(allGroups)
    filePaths = allGroups{g}; % Get file paths for the current group
    nChannels = length(filePaths); % Number of channels

    if nChannels < 2
        warning('Skipping %s: Requires at least 2 channels.', groupNames{g});
        continue;
    end

    % Step 2: Read Audio Data from Files
    chunkSize = 1e6; % Number of samples per chunk (adjust based on memory availability)
    audioData = zeros(chunkSize, nChannels, 'single'); % Preallocate for a chunk

for i = 1:nChannels
    fileID = fopen(filePaths{i}, 'r'); % Open the file for reading
    totalSamples = 0;

    while ~feof(fileID)
        % Read a chunk of data
        data = fread(fileID, [chunkSize, 1], 'single');
        numSamples = length(data);

        % Dynamically expand the array if needed
        if totalSamples + numSamples > size(audioData, 1)
            audioData = [audioData; zeros(chunkSize, nChannels, 'single')]; %#ok<AGROW>
        end

        % Add data to the channel
        audioData(totalSamples + (1:numSamples), i) = data;
        totalSamples = totalSamples + numSamples;
    end

    fclose(fileID); % Close the file
end

 

    % Step 3: Microphone Array Geometry
    array = phased.ULA('NumElements', nChannels, 'ElementSpacing', micSpacing);

    % Step 4: Create the Beamformer
    targetDirection = [0; 0]; % Target azimuth and elevation (broadside)
    beamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
        'SampleRate', fs, ...
        'PropagationSpeed', c, ...
        'Direction', targetDirection);

    % Step 5: Apply Beamforming
    beamformedSignal = beamformer(audioData);

    % Step 6: Save the Beamformed Signal
    outputFileName = sprintf('%s_beamformed_output.wav', groupNames{g});
    outputFilePath = fullfile(outputFolder, outputFileName);
    audiowrite(outputFilePath, beamformedSignal, fs);

    % Display Progress
    fprintf('Beamformed output saved to %s\n', outputFilePath);
end
