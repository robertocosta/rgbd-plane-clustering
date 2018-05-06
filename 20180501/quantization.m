
function q = quantization(im, from, to, quantizationLevels)
    im(isnan(im)) = 0;
    qVec = from:(to-from)/quantizationLevels:to;
    qVec = qVec(2:end-1);
%     thresh = multithresh(im,glob.quantizationLevels);
%     thresh = min(im(:)):(max(im(:))-min(im(:)))/(glob.quantizationLevels):max(im(:));
%     thresh = thresh(2:end);
    q = imquantize(im,qVec);
end