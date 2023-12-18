function ycbcr_image = ycbcr(rgb_image)
    % Transformación de RGB a YCbCr
    R = rgb_image(:, :, 1);
    G = rgb_image(:, :, 2);
    B = rgb_image(:, :, 3);
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B;
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B;
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B;
    
    ycbcr_image = cat(3, Y, Cb, Cr);
end