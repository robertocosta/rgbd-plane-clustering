function imDepth = rgb_p2rgb_w(imgDepth)
%rgb_p2rgb_w(imgDepth) gets the 3D coordinates from depth and camera params
fx_rgb = 5.1885790117450188e+02;
fy_rgb = 5.1946961112127485e+02;
cx_rgb = 3.2558244941119034e+02;
cy_rgb = 2.5373616633400465e+02;
  [H, W] = size(imgDepth);

  % Make the original consistent with the camera location:
  [xx, yy] = meshgrid(1:W, 1:H);

  x3 = (xx - cx_rgb) .* imgDepth / fx_rgb;
  y3 = (yy - cy_rgb) .* imgDepth / fy_rgb;
  z3 = imgDepth;
  imDepth = cat(3,x3,z3,-y3+max(y3(:)));
  %points3d = [x3(:) -y3(:) z3(:)];
end