scenario = test_env_v1;
%scenario = drivingScenarioTrafficExample;
% Default car properties

carLen   = 4.7; % in meters
carWidth = 1.8; % in meters
rearAxleRatio = .25;

% Define road dimensions

laneWidth   = carWidth*2; % in meters

%scenario.Simulat
ionTime = 40;

plot(scenario);

%% constructing Refernce path
a_star_path = xlsread("refpath_tr3.xls");
waypoints = a_star_path(1:5:end,1:2);
%waypoints = a_star_path;

% waypoints = [0 50; 150 50; 300 75; 310 75; 400 0; 300 -50; 290 -50; 0 -50]; % in meters
refPath = referencePathFrenet(waypoints);

fprintf('refPath : %s',class(refPath));

figure(2);
show(refPath);

ax = show(refPath);
axis(ax,'equal'); xlabel('X'); ylabel('Y');

%% contructing Trajectory gnerator

connector = trajectoryGeneratorFrenet(refPath);

%% constructing dynamic collision checker
capList = dynamicCapsuleList;

%Creating a geometry structure for the ego vehicle with the given parameters.
egoID = 1;
[egoID, egoGeom] = egoGeometry(capList,egoID);

egoGeom.Geometry.Length = carLen; % in meters
egoGeom.Geometry.Radius = carWidth/2; % in meters
egoGeom.Geometry.FixedTransform(1,end) = -carLen*rearAxleRatio; % in meters


%adding the ego vechile to the dynamic collision list
updateEgoGeometry(capList,egoID,egoGeom);


actorID = (1:5)';
actorGeom = repelem(egoGeom,5,1);
updateObstacleGeometry(capList,actorID,actorGeom)

%% defining simulator parameters
% Synchronize the simulator's update rate to match the trajectory generator's
% discretization interval.
scenario.SampleTime = connector.TimeResolution; % in seconds

% Define planning parameters.
replanRate = 10; % Hz

% Define the time intervals between current and planned states.
%timeHorizons = 1:3; % in seconds
% maxHorizon = max(timeHorizons); % in seconds to 1:5

%increasing the horizon time
timeHorizons = 1:3; % in seconds
maxTimeHorizon = max(timeHorizons); % in seconds to 1:5

%increasing the MaxNUmSTep according to the timeHorizon
capList.MaxNumSteps = 1+floor(maxTimeHorizon/scenario.SampleTime);
%capList.MaxNumSteps = 51;

% Define cost parameters.
latDevWeight    =  1;
timeWeight      = -1;
speedWeight     =  1;

% Reject trajectories that violate the following constraints.
maxAcceleration =  15; % in meters/second^2
maxCurvature    =   1; % 1/meters, or radians/meter
minVelocity     =   0; % in meters/second

% Desired velocity setpoint, used by the cruise control behavior and when
% evaluating the cost of trajectories.
speedLimit = 11; % in meters/second

% Minimum distance the planner should target for following behavior.
safetyGap = 10; % in meters


%% Initializing the simulator

[scenarioViewer,futureTrajectory,actorID,actorPoses,egoID,egoPoses,stepPerUpdate,egoState,isRunning,lineHandles] = exampleHelperInitializeSimulator(scenario,capList,refPath,laneWidth,replanRate,carLen);

%% running driving simulation

tic
while isRunning
    % Retrieve the current state of actor vehicles and their trajectory over
    % the planning horizon.
    [curActorState,futureTrajectory,isRunning] = exampleHelperRetrieveActorGroundTruth(scenario,futureTrajectory,replanRate,maxTimeHorizon);
    fprintf('isRunning : %d\n',isRunning);
    % Generate cruise control states.
    [termStatesCC,timesCC] = exampleHelperBasicCruiseControl(refPath,laneWidth,egoState,speedLimit,timeHorizons);
    
    % Generate lane change states.
    [termStatesLC,timesLC] = exampleHelperBasicLaneChange(refPath,laneWidth,egoState,timeHorizons);
    
    % Generate vehicle following states.
     [termStatesF,timesF] = exampleHelperBasicLeadVehicleFollow(refPath,laneWidth,safetyGap,egoState,curActorState,timeHorizons);
    
    % Combine the terminal states and times.
    allTS = [termStatesCC; termStatesLC; termStatesF];
    allDT = [timesCC; timesLC; timesF];
    numTS = [numel(timesCC); numel(timesLC); numel(timesF)];

%     allTS = [termStatesCC; termStatesLC];
%     allDT = [timesCC; timesLC];
%     numTS = [numel(timesCC); numel(timesLC)];
    
    % Evaluate cost of all terminal states.
    costTS = exampleHelperEvaluateTSCost(allTS,allDT,laneWidth,speedLimit,speedWeight, latDevWeight, timeWeight);
    % Generate trajectories.
    egoFrenetState = global2frenet(refPath,egoState);
    [frenetTraj,globalTraj] = connect(connector,egoFrenetState,allTS,allDT);
    
    % Eliminate trajectories that violate constraints.
    isValid = exampleHelperEvaluateTrajectory(globalTraj,maxAcceleration,maxCurvature,minVelocity);
    % Update the collision checker with the predicted trajectories
    % of all actors in the scene.
    for i = 1:numel(actorPoses)
        actorPoses(i).States = futureTrajectory(i).Trajectory(:,1:3);
    end
    updateObstaclePose(capList,actorID,actorPoses);
    
    % Determine evaluation order.
    [cost, idx] = sort(costTS);
    optimalTrajectory = [];
    
    trajectoryEvaluation = nan(numel(isValid),1);
    
    % Check each trajectory for collisions starting with least cost.
    for i = 1:numel(idx)
        if isValid(idx(i))
            % Update capsule list with the ego object's candidate trajectory.
            egoPoses.States = globalTraj(idx(i)).Trajectory(:,1:3);
            updateEgoPose(capList,egoID,egoPoses);
            
            % Check for collisions.
            isColliding = checkCollision(capList);
            
            if all(~isColliding)
                % If no collisions are found, this is the optimal.
                % trajectory.
                trajectoryEvaluation(idx(i)) = 1;
                optimalTrajectory = globalTraj(idx(i)).Trajectory;
                break;
            else
                trajectoryEvaluation(idx(i)) = 0;
            end
        end
    end
    % Display the sampled trajectories.
    lineHandles = exampleHelperVisualizeScene(lineHandles,globalTraj,isValid,trajectoryEvaluation);

    hold on;
    show(capList,'TimeStep',1:capList.MaxNumSteps,'FastUpdate',1);
    hold off;
    
    if isempty(optimalTrajectory)
        % All trajectories either violated a constraint or resulted in collision.
        %
        %   If planning failed immediately, revisit the simulator, planner,
        %   and behavior properties.
        %
        %   If the planner failed midway through a simulation, additional
        %   behaviors can be introduced to handle more complicated planning conditions.
        error('No valid trajectory has been found.');
    else
        % Visualize the scene between replanning.
        for i = (2+(0:(stepPerUpdate-1)))
            % Approximate realtime visualization.
            dt = toc;
            if scenario.SampleTime-dt > 0
                pause(scenario.SampleTime-dt);
            end
            
            egoState = optimalTrajectory(i,:);
            scenarioViewer.Actors(1).Position(1:2) = egoState(1:2);
            scenarioViewer.Actors(1).Velocity(1:2) = [cos(egoState(3)) sin(egoState(3))]*egoState(5);
            scenarioViewer.Actors(1).Yaw = egoState(3)*180/pi;
            scenarioViewer.Actors(1).AngularVelocity(3) = egoState(4)*egoState(5);
            
            % Update capsule visualization.
            hold on;
            show(capList,'TimeStep',i:capList.MaxNumSteps,'FastUpdate',1);
            hold off;
            
            % Update driving scenario.
            advance(scenarioViewer);
            tic;
        end
    end
end
