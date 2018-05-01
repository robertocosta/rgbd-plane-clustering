
function q = quantization(im)
global glob;
    im(isnan(im)) = 0;
%     thresh = multithresh(im,glob.quantizationLevels);
%     thresh = min(im(:)):(max(im(:))-min(im(:)))/(glob.quantizationLevels):max(im(:));
%     thresh = thresh(2:end);
    q = imquantize(im,glob.quantizationLevels);
end