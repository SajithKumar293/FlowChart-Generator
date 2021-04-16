close all
clc
clear

% Read Image and Initialise output directory
original = imread('./fc.jpg'); 
outFolder = './Outputs/test';
mkdir(outFolder);
figure;imshow(original);
title('Original Image');

% Convert the image to grayscale and then resize
ImgGrey = rgb2gray(original);
[rows, cols] = size(ImgGrey);
nrows = rows/10; 
ncols = cols/10;
ImgResized = imresize(ImgGrey, [nrows, ncols]);

% Adaptive Thresholding and Binarization
ImgBinary = imbinarize(ImgResized, adaptthresh(ImgResized, 0.3, 'ForegroundPolarity','dark'));
figure;imshow(ImgBinary);
title('Binary Image');
outFile = 'BinaryImage.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

% Invert the Image
ImgInverted = 1-ImgBinary;
figure;imshow(ImgInverted);
title('Inverted Image');
outFile = 'InvertedImage.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

% Remove noise at corners
CC = bwconncomp(ImgInverted);
S = regionprops(CC, 'Area');
L = labelmatrix(CC);
ImgCleaned = ismember(L, find([S.Area] >= nrows/2));
figure;imshow(ImgCleaned);
title('Image Cleaned');
outFile = 'CleanedImage.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

% Fill the holes in the  image
ImgFilled = imfill(ImgCleaned, 'holes');
figure;imshow(ImgFilled);
title('Filled Image');
outFile = 'FilledImage.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

% Zero Cross Edge Detection
ImgEdge = edge(ImgFilled, 'zerocross');
figure;imshow(ImgEdge);
title('Edge Detection');
outFile = 'EdgeDetection.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

% Hough Transform
[H,theta,rho] = hough(ImgEdge); 
peaks = houghpeaks(H, 100);
figure;imshow(H,[],'XData',theta,'YData',rho,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
plot(theta(peaks(:,2)),rho(peaks(:,1)),'s','color','white');
colormap(gca, hot);

hlines = houghlines(ImgEdge,theta,rho,peaks,'FillGap',5,'MinLength',2); 
figure, imshow(ImgCleaned), hold on
max_len = 0;
for k = 1:length(hlines)
   xy = [hlines(k).point1; hlines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   len = norm(hlines(k).point1 - hlines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
title('Detected Lines');
outFile = 'HoughTransform.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

% Predict the best angle to rotate and rotate 
hlines = houghlines(ImgEdge, theta, rho, peaks);
bestAngle = mode([hlines.theta])+90;
ImgRotated = imrotate(ImgCleaned, bestAngle);
ImgFilledRotated = imrotate(ImgFilled, bestAngle);
figure;imshow(ImgRotated);
title('Rotated Image');
outFile = 'RotatedImage.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

[nrows, ncols] = size(ImgRotated);
figure;imshow(ImgFilledRotated);
title('Filled Image');
outFile = 'RotatedFilledImage.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

% Decomposition of flowchart
se = strel('diamond', 5);
ImgOpened = imopen(ImgFilledRotated, se);
Imgbw = bwareaopen(ImgOpened, 50);
ImgArray = ImgRotated - Imgbw;
ImgArray = imbinarize(ImgArray);

ImgArrows = bwareaopen(ImgArray, 20); 
CC_arrows = bwconncomp(ImgArrows);
S_arrows = regionprops(CC_arrows, 'Area');
L_arrows = labelmatrix(CC_arrows);
ImgArrows_new = ismember(L_arrows, find([S_arrows.Area] >= nrows/10));

figure; imshow(ImgArrows_new);
ImgArrows = ImgArrows_new;
title('Only Arrows');
outFile = 'Arrows.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

ImgShapes = ImgRotated - ImgArrows; 
figure;imshow(ImgShapes);
title('Only Shapes');
outFile = 'Shapes.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

% Decompose Shapes into Circles, Rectangles and Diamonds
[shapeLbl, n_shapeLbl] = bwlabel(ImgShapes); 
figure; imagesc(shapeLbl); axis equal;
title('Different Shapes in Image');
outFile = 'Different Shapes.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

shapeProps = regionprops(shapeLbl, 'all'); 
centroidShapes = cat(1, shapeProps.Centroid);
perimeterShapes = cat(1, shapeProps.Perimeter); 
areaShapes = cat(1, shapeProps.ConvexArea);
bbShapes = cat(1, shapeProps.BoundingBox);
areaRatioCircle = (perimeterShapes.^2)./(4*pi*areaShapes);
areaRatioRect = NaN(n_shapeLbl,1); 

for i = 1:n_shapeLbl
    [p,q] = size(shapeProps(i).FilledImage); 
    areaRatioRect(i) = areaShapes(i)/(p*q);
end

isCircle = (areaRatioCircle < 1.1);
isRect = (areaRatioRect > 0.75); 
isRect = logical(isRect .* ~isCircle); 
isDiamond = (areaRatioRect <= 0.75);
isDiamond = logical(isDiamond .* ~isCircle);

% Find Arrow Orientation, Arrow Head and Arrow Tail
[arrowLabels, n_arrowLabels] = bwlabel(ImgArrows); 
figure; imagesc(arrowLabels); axis equal;
title('Different Arrows in Image');
outFile = 'Different Arrows.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

arrowProps = regionprops(arrowLabels, 'all'); 
arrowCentroids = cat(1, arrowProps.Centroid); 
arrowBBs = cat(1, arrowProps.BoundingBox); 
arrowCentres = [arrowBBs(:, 1) + 0.5*arrowBBs(:, 3), arrowBBs(:, 2) + 0.5*arrowBBs(:, 4)];
figure; imshow(ImgArrows);

hold on;
plot(arrowCentres(:, 1), arrowCentres(:, 2), 'r*', 'LineWidth', 2, 'MarkerSize', 5);
plot(arrowCentroids(:, 1), arrowCentroids(:, 2), 'b*', 'LineWidth', 2, 'MarkerSize', 5);
title('Arrow Centres and Centroids');
outFile = 'ArrowCentresCentroids.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

% Find head and tail of the arrow based on its orientation
arrowBBsMidpts = [];
allArrowHeads = []; 
allArrowTails = []; 

for i = 1:n_arrowLabels
    hold on;
    arrowOrient = arrowProps(i).Orientation; 
    if (abs(abs(arrowOrient)-90) > abs(arrowOrient))
        arrowBBMidpt = [arrowBBs(i, 1), arrowCentres(i, 2);  arrowBBs(i, 1) + arrowBBs(i, 3), arrowCentres(i, 2)];
    else 
        arrowBBMidpt = [arrowCentres(i, 1), arrowBBs(i, 2); arrowCentres(i, 1), arrowBBs(i, 2) + arrowBBs(i, 4)];
    end
    
    if (pdist([arrowCentroids(i, :); arrowBBMidpt(1, :)], 'euclidean') <= pdist([arrowCentres(i, :); arrowBBMidpt(1, :)], 'euclidean'))
        arrowHead = arrowBBMidpt(1, :);
        arrowTail = arrowBBMidpt(2, :);
    else
        arrowHead = arrowBBMidpt(2, :);
        arrowTail = arrowBBMidpt(1, :);
    end
    
    plot(arrowHead(:, 1), arrowHead(:, 2), 'g*', 'LineWidth', 2, 'MarkerSize', 5);
    plot(arrowTail(:, 1), arrowTail(:, 2), 'y*', 'LineWidth', 2, 'MarkerSize', 5);
    
    arrowBBsMidpts = [arrowBBsMidpts; arrowBBMidpt];
    allArrowHeads = [allArrowHeads; arrowHead];
    allArrowTails = [allArrowTails; arrowTail];
end
title('Arrows Heads and Tails');
outFile = 'ArrowsHeadTail.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

% Find Closest Shape to Arrow Head
bbShapesMidpts = [];

for i = 1:n_shapeLbl 
    shapeBB = shapeProps(i).BoundingBox; 
    shapeBBMidpt1 = [shapeBB(1) + 0.5*shapeBB(3), shapeBB(2)];
    shapeBBMidpt2 = [shapeBB(1) + shapeBB(3), shapeBB(2) + 0.5*shapeBB(4)];
    shapeBBMidpt3 = [shapeBB(1) + 0.5*shapeBB(3), shapeBB(2) + shapeBB(4)];
    shapeBBMidpt4 = [shapeBB(1), shapeBB(2) + 0.5*shapeBB(4)];
    bbShapesMidpts = [bbShapesMidpts; shapeBBMidpt1; shapeBBMidpt2; shapeBBMidpt3; shapeBBMidpt4];
end

figure;imshow(ImgShapes);
hold on;
plot(bbShapesMidpts(:, 1), bbShapesMidpts(:, 2), 'r*');
title('Shapes Anchors');
outFile = 'ShapeAnchors.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);

arrowHeads = []; 
arrowTails = [];

for i = 1:size(allArrowHeads, 1)
    arr_shape_dists = [];
    
    for j = 1:size(bbShapesMidpts, 1)
        arr_shape_dist = pdist([allArrowHeads(i, 1), allArrowHeads(i, 2); bbShapesMidpts(j, 1), bbShapesMidpts(j, 2)],'euclidean');
        arr_shape_dists = [arr_shape_dists; arr_shape_dist];
    end
    
    [~, minidx] = min(arr_shape_dists(:)); 
    arrowHeads = [arrowHeads;  bbShapesMidpts(minidx, :)];
    arrowTails = [arrowTails; allArrowTails(i, :)];
end

% Plot final arrow heads and tails
hold on;
plot(arrowHeads(:, 1), arrowHeads(:, 2), 'b*');
plot(arrowTails(:, 1), arrowTails(:, 2), 'y*');
title('Final Arrows Heads and Tails');
outFile = 'FinalArrowsHeadTail.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);


% Circles
finalIm = ones(nrows, ncols);
figure; imshow(finalIm);
circleCentres = centroidShapes(isCircle,:); %Centre of each circle
circleRadii = perimeterShapes(isCircle,:)./(2*pi); %Radius of each circle
viscircles(circleCentres, circleRadii, 'Color', 'k');

% Arrows
arrow('Start', arrowTails, 'Stop', arrowHeads, 'Type', 'line', 'LineWidth', 2);

% Rectangles
rectsBBs = bbShapes(isRect, :);

for i = 1:size(rectsBBs, 1)
    rectangle('Position', [rectsBBs(i,1) rectsBBs(i,2)...
        rectsBBs(i,3) rectsBBs(i,4)], 'EdgeColor','k',...
    'LineWidth',3);
    hold on;
end

% Diamonds
hold on;
diadsBBs = bbShapes(isDiamond, :);

for i = 1:size(diadsBBs,1)
    patch([diadsBBs(i,1)+ 0.5*diadsBBs(i,3) diadsBBs(i,1)+diadsBBs(i,3) ...
        diadsBBs(i,1)+0.5*diadsBBs(i,3) diadsBBs(i,1) ],...
        [diadsBBs(i,2) diadsBBs(i,2)+0.5*diadsBBs(i,4) ...
        diadsBBs(i,2)+diadsBBs(i,4) diadsBBs(i,2)+0.5*diadsBBs(i,4) ], 'w', 'EdgeColor', 'k', 'LineWidth',3);
    hold on;
end

% Save Final Image
title('Output Flowchart');
outFile = 'OutputImage.jpg';
outPath = fullfile(outFolder, outFile);
saveas(gcf, outPath);
