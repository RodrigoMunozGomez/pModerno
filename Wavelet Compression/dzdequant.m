function vector = dzdequant(q,step)
    % if channel == 1
    %     if level >= 3
    %         energy = 1;
    %     elseif level == 2
    %         if subband == "HL" || subband == "LH"
    %            energy = 0.998276;
    %         elseif subband == "HH"
    %             energy = 0.996555;
    %         end
    %     elseif level == 1
    %         if subband == "HL" || subband == "LH"
    %            energy = 0.756353;
    %         elseif subband == "HH"
    %             energy = 0.573057;
    %         end        
    %     end
    % end
    % step = step/energy;
    vector = q.*step;
end