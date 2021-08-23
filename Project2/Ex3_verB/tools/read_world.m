function landmarks = read_world(filename)
    % Reads the world definition and returns a structure of landmarks.
    %
    % filename: path of the file to load
    % landmarks: structure containing the parsed information
    %
    % Each landmark contains the following information:
    % - id : id of the landmark
    % - x  : x-coordinate
    % - y  : y-coordinate
    %
    % Examples:
    % - Obtain x-coordinate of the 5-th landmark
    %   landmarks(5).x
    input = fopen(filename);

%     landmarks = struct;
     line=3;
     i=1;
    while sum(line)>0
        line = fgetl(input);
        try
        data = strsplit(line, ' ');

%         landmark = struct('id', str2double(data{1}),...
%             'x' , str2double(data{2}),...
%             'y' , str2double(data{3}));
%         landmarks{i+1} = landmark;
        landmarks(i,:)=[str2double(data{1}) str2double(data{2}) str2double(data{3})];
        i=1+i;
        end
    end

%     landmarks = landmarks(2:end);

    fclose(input);
end
