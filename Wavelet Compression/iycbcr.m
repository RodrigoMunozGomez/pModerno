function rgb_image = iycbcr(ycbcr_image)
    % Transformación inversa de YCbCr a RGB
    Y = ycbcr_image(:, :, 1);
    Cb = ycbcr_image(:, :, 2);
    Cr = ycbcr_image(:, :, 3);
    
    R = Y + 1.402 * Cr;
    G = Y - 0.3441 * Cb - 0.7141 * Cr;
    B = Y + 1.772 * Cb;
    
    rgb_image = cat(3, R, G, B);
end