function plot_state(gt, preds, mu, sigma, landmarks, timestep, observedLandmarks, z, window)
    % Visualizes the state of the EKF SLAM algorithm.
    %
    % The resulting plot displays the following information:
    % - map ground truth (black +'s)
    % - current robot pose estimate (red)
    % - current landmark pose estimates (blue)
    % - visualization of the observations made at this time step (line between robot and landmark)

    clf;
    hold on
    grid on
    L = (landmarks); 
    drawprobellipse(mu(1:3), sigma(1:3,1:3), 0.6, 'r');
    h(1) = plot(L(:,2), L(:,3), 'k+', 'markersize', 5, 'linewidth', 2, 'DisplayName', 'landmarks');
    for i=1:length(observedLandmarks)
        if(observedLandmarks(i))
            h(2) = plot(mu(2*i+ 2),mu(2*i+ 3), 'bo', 'markersize', 5, 'linewidth', 2);
            drawprobellipse(mu(2*i+ 2:2*i+ 3), sigma(2*i+ 2:2*i+ 3,2*i+ 2:2*i+ 3), 0.6, 'b');
        end
    end

    for i=1:size(z,2)
        mX = mu(2*z(i).id+2);
        mY = mu(2*z(i).id+3);
        line([mu(1), mX],[mu(2), mY], 'color', 'k', 'linewidth', 1);
    end

    drawrobot(mu(1:3), 'r', 3, 0.3, 0.3);
    
    h(3) = plot(gt(1,1:end),gt(2,1:end), 'k-', 'markersize', 5, 'linewidth', 2);
    h(4) = plot(preds(1,1:end),preds(2,1:end), 'c-', 'markersize', 5, 'linewidth', 2);
        
        
    xlim([-2, 12])
    ylim([-2, 12])
    hold off
    
    legend(h([1 2 3 4]),'GT - landmarks', 'Est - landmarks', 'GT - path', 'Est - path')
    
    if 1 %window
    hold on
      drawnow;
      pause(0.1);
    else
%       figure(1, 'visible', 'off');
%       filename = sprintf('../plots/ekf_%03d.png', timestep);
%       print(filename, '-dpng');
    end
end
