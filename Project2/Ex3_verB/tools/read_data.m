function data = read_data(filename)
    % Reads the odometry and sensor readings from a file.
    %
    % filename: path to the file to parse
    % data: structure containing the parsed information
    %
    % The data is returned in a structure where the u_t and z_t are stored
    % within a single entry. A z_t can contain observations of multiple
    % landmarks.
    %
    % Usage:
    % - access the readings for timestep i:
    %   data.timestep(i)
    %   this returns a structure containing the odometry reading and all
    %   landmark obsevations, which can be accessed as follows
    % - odometry reading at timestep i:
    %   data.timestep(i).odometry
    % - senor reading at timestep i:
    %   data.timestep(i).sensor
    %
    % Odometry readings have the following fields:
    % - r1 : rotation 1
    % - t  : translation
    % - r2 : rotation 2
    % which correspond to the identically labeled variables in the motion
    % mode.
    %
    % Sensor readings can again be indexed and each of the entris has the
    % following fields:
    % - id      : id of the observed landmark
    % - range   : measured range to the landmark
    % - bearing : measured angle to the landmark (you can ignore this)
    %
    % Examples:
    % - Translational component of the odometry reading at timestep 10
    %   data.timestep(10).odometry.t
    % - Measured range to the second landmark observed at timestep 4
    %   data.timestep(4).sensor(2).range
    input = fopen(filename);

    data = struct;
    data.timestep.sensor = struct;
    first = 0;

    odom = struct;
    sensor = struct;
    T=0;
    line=1;
    while line>0
        line = fgetl(input);
        if line<0
            return
        end
        arr = strsplit(line, ' ');
        type = deblank(arr{1});

        if(strcmp(type, 'ODOMETRY') == 1)
            if(first == 1)
                data.timestep(T).odometry = odom;
                data.timestep(T).sensor = sensor;
                odom = struct;
                sensor = struct;
            end
            first = 1;
            T=T+1;
            odom.r1 = str2double(arr{2});
            odom.t  = str2double(arr{3});
            odom.r2 = str2double(arr{4});
            ind=1;
            k=0;
           % sensor= struct;
        elseif(strcmp(type, 'SENSOR') == 1)
            sensor(ind+k).id      = str2double(arr{2});
            sensor(ind+k).range  = str2double(arr{3});
            sensor(ind+k).bearing = str2double(arr{4});
            k=k+1;
        end
    end

 %   data.timestep = data.timestep(2:end);

    fclose(input);
end
