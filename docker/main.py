import kagglehub
import sklearn

# Download latest version
path = kagglehub.dataset_download("varpit94/ethereum-data")

print("Path to dataset files:", path)