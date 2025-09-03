import os
import shutil
def prepare_label_directory(X_test, y_test):
    # Storing the test labels in different directory
    # Combine X_test and y_test into one DataFrame

    combined = X_test.copy()
    combined['label'] = y_test.values  # Make sure index alignment is preserved

    # Create output folder if not exists
    output_dir = "/Users/farhat/Documents/Project/ProcessedData/LabelDirectory"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Group by patient_id and save each group as a separate .psv file
    for patient_id, group in combined.groupby('Patient_Id'):
        # Only save the label column (or keep other columns if needed)
        patient_labels = group[['label']].rename(columns={'label': 'SepsisLabel'})
        
        # Build file path
        filename = f"{patient_id}.psv"
        filepath = os.path.join(output_dir, filename)
        
        # Save to .psv (pipe-separated)
        patient_labels.to_csv(filepath, sep='|', index=False)
    return combined, output_dir
def prepare_prediction_directory (combined, y_pred):
    # Add predicted probabilities
    combined['PredictedProbability'] = y_pred

    # Add predicted labels based on threshold 0.5
    combined['PredictedLabel'] = (combined['PredictedProbability'] >= 0.5).astype(int)
    # Ensure patient_id is present
    if 'Patient_Id' not in combined.columns:
        raise ValueError("The column 'patient_id' must be present in X_test.")

    # Create output folder
    output_dir = "/Users/farhat/Documents/Project/ProcessedData/PredictionDirectory"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Group by patient_id and write predictions
    for patient_id, group in combined.groupby('Patient_Id'):
        # Keep only necessary columns
        patient_predictions = group[['PredictedLabel', 'PredictedProbability']]

        # Create file path
        filename = f"{patient_id}.psv"
        filepath = os.path.join(output_dir, filename)

        # Save to .psv
        patient_predictions.to_csv(filepath, sep='|', index=False)
    return output_dir