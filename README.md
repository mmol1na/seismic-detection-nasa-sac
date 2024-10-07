# Seismic Event Detection Using MiniSEED Files

This repository contains code and data for detecting seismic events using MiniSEED (.mseed) files, focusing on lunar seismic test data. The project utilizes a machine learning model to identify seismic events and generate a catalog of detected events. The codebase includes utilities to process, analyze, and visualize seismic signals, providing an end-to-end framework for seismic event detection.

This project is part of the **2024 NASA Space Apps Challenge**, specifically addressing the challenge titled **Seismic Detection Across the Solar System**.

## Overview

The project is built to detect seismic events in MiniSEED files, which are a common format for storing seismic signal data. Using a classification model trained on labeled seismic data, the system identifies seismic events in new data. The pipeline includes components for data loading, preprocessing, model training, prediction, and visualization.

### Components

#### 1. **MSEEDLoader**

- **Purpose**: Loads and preprocesses MiniSEED files by normalizing the seismic signal data (subtracting the mean and dividing by the standard deviation). It extracts essential information such as time series, sampling rate, and the start time of the recording.
- **Usage**: Handles input MiniSEED files for both training and prediction phases.

#### 2. **TrainingDataPreparer**

- **Purpose**: Prepares data for model training by loading labeled MiniSEED files and event labels. It selects time windows that contain seismic events, aligns the data with the labels, and constructs feature vectors (`X`) and corresponding labels (`y`) for training.
- **Process**: Ensures proper alignment between events and the seismic data, and balances the training dataset.

#### 3. **QuakeModel**

- **Purpose**: Implements a Random Forest classifier to detect seismic events. It trains the model on the prepared data and evaluates its performance using accuracy metrics and classification reports.
- **Features**:
  - Trains a classifier with seismic event data.
  - Evaluates performance with metrics like accuracy and confusion matrix.
  - Visualizes classification results using a confusion matrix heatmap.

#### 4. **QuakeDetector**

- **Purpose**: Utilizes the trained model to detect seismic events in new test MiniSEED files. It processes the files, predicts seismic activity, and records the detected events in a catalog (`quake_catalog.csv`).
- **Features**:
  - Processes new MiniSEED data and detects potential quakes.
  - Logs all detections to a catalog, including filename, absolute and relative timestamps, unique event IDs, and event types (e.g., `impact_mq`).
  - Generates graphs for each detected event, visualizing the filtered seismic signals.

#### 5. **DataPreprocessor**

- **Purpose**: Splits seismic data from new MiniSEED files into appropriate segments for prediction, aligning the data with the model's training configuration.

### Outputs

- **Catalog of Detected Events**: A CSV file (`quake_catalog.csv`) is generated, listing all detected seismic events. Each entry includes:
  - `filename`: The name of the MiniSEED file.
  - `time_abs(%Y-%m-%dT%H:%M:%S.%f)`: The absolute time of the event.
  - `time_rel(sec)`: The time in which the window that contains the event is, not the specific event time.
  - `evid`: A unique event identifier.
  - `mq_type`: The event type, such as `impact_mq`. (since recognition was not required impact_mq is hardcoded)
- **Filtered Signal Graphs**: For each detected event, a graph is generated, displaying the filtered seismic signal. These graphs allow for visual analysis of detected seismic events.

## Model Performance

The Random Forest classifier achieves a certain level of accuracy, reflecting its ability to distinguish seismic events from noise. The confusion matrix visualization provides insights into the model's performance across different classes of events.
