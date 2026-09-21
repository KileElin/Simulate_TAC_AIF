import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

# Load the CSV file
def load_data(file_path):
    """
    Load the CSV file with ';' as the delimiter and return the data as a Pandas DataFrame.
    """
    data = pd.read_csv(file_path, sep=';')  # Specify the delimiter as ';'
    return data

# Shift and log-transform the data
def preprocess_data(data):
    """
    Shift the data to make all values positive and apply log transformation.
    """
    # Exclude the first column (patient_id)
    parameters = data.iloc[:, 1:]
    
    # Find the minimum value in the data
    min_value = parameters.min().min()  # Minimum value across all columns
    
    # Shift the data to make all values positive
    shift_value = abs(min_value) + 1 if min_value <= 0 else 0
    parameters = parameters + shift_value
    
    # Apply log transformation
    log_transformed = np.log(parameters)
    
    return log_transformed, shift_value

  
# Calculate mean and covariance
def calculate_distribution_parameters(data):
    """
    Calculate the mean vector and covariance matrix from the data.
    """
    # Convert the DataFrame to a NumPy array
    parameters = data.to_numpy()
    
    # Calculate the mean vector (1D array)
    mean_vector = np.mean(parameters, axis=0)
    
    # Calculate the covariance matrix (2D array)
    covariance_matrix = np.cov(parameters, rowvar=False)
    
    return mean_vector, covariance_matrix
  
# Create a multivariate normal distribution and draw samples with constraints
def generate_samples_with_constraints(mean_vector, covariance_matrix, shift_value, num_samples=10, fix_vB=False):
    """
    Generate samples from a multivariate normal distribution and enforce constraints.
    If fix_vB=True, set vB to 0.02 for all samples.
    """
    # Regularize the covariance matrix to ensure it is positive definite
    epsilon = 1e-6
    covariance_matrix += np.eye(covariance_matrix.shape[0]) * epsilon
    
    # Create the multivariate normal distribution
    mvn = multivariate_normal(mean=mean_vector, cov=covariance_matrix)
    
    # Generate samples and enforce constraints
    valid_samples = []
    while len(valid_samples) < num_samples:
        sample = mvn.rvs()  # Generate a single sample
        
        # Reverse the log transformation and shifting
        sample_original = np.exp(sample) - shift_value
        
        # Apply constraints
        if (
            sample_original[1] > 0 and  # A1m > 0
            sample_original[2] > 0 and  # A2m > 0
            sample_original[3] > 0 and  # A3m > 0
            sample_original[4] < 0 and  # L1m < 0
            sample_original[5] < 0 and  # L2m < 0
            sample_original[6] < 0 and  # L3m < 0
            sample_original[7] > 0 and  # vB > 0
            sample_original[8] > 0 and  # K1 > 0
            sample_original[9] > 0 and  # k2 > 0
            sample_original[10] > 0   # k3 > 0
            #sample_original[11] > 0  # k4 > 0
        ):
            # If fix_vB is enabled, set vB to 0.02
            if fix_vB:
                sample_original[7] = 0.02  # Set vB to 0.02
            
            valid_samples.append(sample_original)
    
    return np.array(valid_samples)
  
# Reverse the transformation
def reverse_transformation(log_transformed_data, shift_value):
    """
    Reverse the log transformation and shifting to recover the original data.
    """
    # Exponentiate the log-transformed data
    exp_data = np.exp(log_transformed_data)
    
    # Undo the shift
    original_data = exp_data - shift_value
    
    return original_data
  
# Main function 
def main(file_path, output_file, num_samples=100, fix_vB=False):
    """
    Main function to load data, preprocess it, calculate distribution parameters, 
    generate samples, reverse the transformation, and save to a CSV file.
    """
    # Load the data
    data = load_data(file_path)
    
    # Ensure the data has at least 12 columns (1 for patient_id + 11 parameters) (13 for muscle/4k model)
    if data.shape[1] != 12:
        raise ValueError("The CSV file must have exactly 12 columns (1 for patient_id and 11 parameters).")
    
    # Preprocess the data (shift and log-transform)
    preprocessed_data, shift_value = preprocess_data(data)
    
    # Calculate the mean vector and covariance matrix
    mean_vector, covariance_matrix = calculate_distribution_parameters(preprocessed_data)
    
    # Generate samples with constraints
    constrained_samples = generate_samples_with_constraints(mean_vector, covariance_matrix, shift_value, num_samples, fix_vB)
    
    # Create a DataFrame for the samples
    column_names = ['sample', 'TAUm', 'A1m', 'A2m', 'A3m', 'L1m', 'L2m', 'L3m', 'vB', 'K1', 'k2', 'k3'] #, 'Ki''k4'
    samples_df = pd.DataFrame(constrained_samples, columns=column_names[1:])  # Exclude 'sample' for now
    samples_df.insert(0, 'sample', range(1, num_samples + 1))  # Add sample numbers
    
    # Save the samples to a CSV file
    samples_df.to_csv(output_file, sep=';', index=False)
    print(f"Generated samples saved to {output_file}")


if __name__ == "__main__":
    # Use the correct file name
    file_path =  "insert_parameter_distribution_file.csv" # Input CSV file
    output_file = "chose_name_of_output_file.csv" # Output CSV file 
    num_samples = 1000  # Number of samples to generate
    fix_vB = False  # Set to True to fix vB to 0.02, or False to generate vB values
    
    # Generate samples and save to CSV
    main(file_path, output_file, num_samples, fix_vB)
