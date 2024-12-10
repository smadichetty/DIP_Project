classdef MorphologicalMatchingApp < handle
    properties
        Figure
        QueryCanvas
        ResultsPanel
        ResultsPanelScrollable
        QueryImagePath
        DatasetImagePaths
        QueryFeatures
        DatasetFeatures
    end
    
    methods
        function app = MorphologicalMatchingApp()
            % Create and configure the main figure
            app.Figure = uifigure('Name', 'Morphological Matching');
            app.Figure.Position = [100, 100, 800, 600];
            
            % Create buttons
            uibutton(app.Figure, 'Text', 'Upload Query Image', ...
                'Position', [150, 550, 150, 30], ...
                'ButtonPushedFcn', @(btn, event) app.uploadQueryImage());
            
            uibutton(app.Figure, 'Text', 'Upload Dataset Images', ...
                'Position', [150, 500, 150, 30], ...
                'ButtonPushedFcn', @(btn, event) app.uploadDatasetImages());
            
            uibutton(app.Figure, 'Text', 'Calculate Similarity', ...
                'Position', [150, 450, 150, 30], ...
                'ButtonPushedFcn', @(btn, event) app.calculateSimilarity());
            
            % Create image display panel
            app.QueryCanvas = uiaxes(app.Figure);
            app.QueryCanvas.Position = [472, 430, 156, 156];
            
            % Create scrollable results panel with 2 columns: 1 for images, 1 for text
            app.ResultsPanel = uipanel(app.Figure, ...
                'Position', [60, 40, 650, 350]);
            app.ResultsPanelScrollable = uigridlayout(app.ResultsPanel, ...
                'Scrollable', 'on', ...
                'RowHeight', {'fit'}, ...
                'ColumnWidth', {'0.5x', '0.5x'});  % Two columns for images and text
        end
        
        function uploadQueryImage(app)
            [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg,*.png,*.bmp)'}); 
            if filename ~= 0
                app.QueryImagePath = fullfile(pathname, filename);
                preprocessedQuery = app.preprocessImage(app.QueryImagePath);
                app.QueryFeatures = app.extractMorphologicalFeatures(preprocessedQuery);
                
                % Display image
                imshow(imread(app.QueryImagePath), 'Parent', app.QueryCanvas);
                % Show success message in the middle of the window
                uialert(app.Figure, 'Query image uploaded and processed successfully!', ...
                    'Success', 'Icon', 'info');
            end
        end
        
        function uploadDatasetImages(app)
            [filenames, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg,*.png,*.bmp)'}, ...
                'Select Dataset Images', 'MultiSelect', 'on');
            if ~isequal(filenames, 0)
                app.DatasetImagePaths = {};
                app.DatasetFeatures = struct();
                
                if ischar(filenames)
                    filenames = {filenames};
                end
                
                for i = 1:length(filenames)
                    fullPath = fullfile(pathname, filenames{i});
                    app.DatasetImagePaths{end + 1} = fullPath;
                    preprocessedImage = app.preprocessImage(fullPath);
                    app.DatasetFeatures.(sprintf('image_%d', i)) = ...
                        app.extractMorphologicalFeatures(preprocessedImage);
                end
                % Show success message in the middle of the window
                uialert(app.Figure, 'Dataset images uploaded and processed successfully!', ...
                    'Success', 'Icon', 'info');
            end
        end
        
        function calculateSimilarity(app)
            if isempty(app.QueryFeatures)
                % Show error message in the middle of the window
                uialert(app.Figure, 'Query features are not extracted. Upload a query image.', ...
                    'Error', 'Icon', 'error');
                return;
            end
            if isempty(app.DatasetFeatures)
                % Show error message in the middle of the window
                uialert(app.Figure, 'Dataset features are not extracted. Upload dataset images.', ...
                    'Error', 'Icon', 'error');
                return;
            end
            
            featurewiseResults = app.calculateFeaturewiseSimilarity();
            app.displayResults(featurewiseResults);
        end
        
        function featurewiseResults = calculateFeaturewiseSimilarity(app)
            % Calculate similarity based on extracted features
            featurewiseResults = struct();
            featureNames = fieldnames(app.QueryFeatures);
            datasetNames = fieldnames(app.DatasetFeatures);
            
            for i = 1:length(datasetNames)
                similarities = zeros(length(featureNames), 1);
                for j = 1:length(featureNames)
                    featureName = featureNames{j};
                    queryFeature = double(app.QueryFeatures.(featureName));
                    datasetFeature = double(app.DatasetFeatures.(datasetNames{i}).(featureName));
                    similarities(j) = norm(queryFeature(:) - datasetFeature(:));
                end
                featurewiseResults.(datasetNames{i}) = similarities;
            end
        end
        
        function displayResults(app, featurewiseResults)
            % Clear previous results
            delete(app.ResultsPanelScrollable.Children);
            
            % Determine the number of dataset images
            numResults = numel(app.DatasetImagePaths);
            
            % Adjust the layout to accommodate all dataset images
            app.ResultsPanelScrollable.RowHeight = repmat({'fit'}, numResults, 1);
            
            % Display new results in grid layout
            datasetNames = fieldnames(featurewiseResults);
            for i = 1:length(datasetNames)
                if i > numResults
                    continue;
                end
                
                % Left column: dataset image
                img = uiimage(app.ResultsPanelScrollable, ...
                    'ImageSource', app.DatasetImagePaths{i}, ...
                    'ScaleMethod', 'fit');
                img.Layout.Row = i;  % Set the row for this image
                img.Layout.Column = 1;  % Set to first column for image
                
                % Right column: similarity results text
                resultText = sprintf('Results for %s:\n', datasetNames{i});
                similarities = featurewiseResults.(datasetNames{i});
                featureNames = fieldnames(app.QueryFeatures);
                
                for j = 1:length(similarities)
                    resultText = sprintf('%s%s: %.2f\n', resultText, featureNames{j}, similarities(j));
                end
                
                % Add a text area for the result text
                resultTextArea = uitextarea(app.ResultsPanelScrollable, ...
                    'Value', resultText, ...
                    'Editable', 'off', ...
                    'FontSize', 12, ...
                    'HorizontalAlignment', 'left');
                resultTextArea.Layout.Row = i;  % Set the row for this text
                resultTextArea.Layout.Column = 2;  % Set to second column for text
            end
        end
    end
    
    methods (Static)
        function enhanced = preprocessImage(imagePath)
            % Read and preprocess image
            img = imread(imagePath);
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            resized = imresize(img, [256, 256]);
            blurred = imgaussfilt(resized, 2);
            enhanced = histeq(blurred);
        end
        
        function features = extractMorphologicalFeatures(image)
            % Extract morphological features
            se = strel('rectangle', [3, 3]); % Adjusted rectangle size
            features = struct();
            features.Erosion = imerode(image, se);
            features.Dilation = imdilate(image, se);
            features.Opening = imopen(image, se);
            features.Closing = imclose(image, se);
        end
    end
end

% Main function to run the application
function runApp()
    app = MorphologicalMatchingApp1();
end
