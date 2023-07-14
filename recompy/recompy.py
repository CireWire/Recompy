import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class Recommender:
    """
    Recommender class for building and training neural networks for recommender systems.
    """

    def __init__(self, columns_to_drop=None):
        """
        Initializes the Recommender class.

        Args:
            columns_to_drop (list): List of column names to drop during data preprocessing. Defaults to None.
        """
        self.model = None
        self.columns_to_drop = columns_to_drop

    def preprocess_data(self, data_file):
        """
        Preprocesses the data by removing unnecessary columns and handling missing values.

        Args:
            data_file (str): Path to the data file in CSV format.

        Returns:
            tuple: A tuple containing train_data, val_data, and test_data pandas DataFrames.
        """
        # Load the data from CSV file
        data = pd.read_csv(data_file)

        # Drop the specified columns if provided
        if self.columns_to_drop:
            data = data.drop(columns=self.columns_to_drop)

        # Handle missing values (NaN) by filling with zeros
        data = data.fillna(0)

        # Split the data into train, validation, and test sets
        train_data = data.sample(frac=0.6, random_state=0)  # 60% for training
        remaining_data = data.drop(train_data.index)
        val_data = remaining_data.sample(frac=0.2, random_state=0)  # 20% for validation
        test_data = remaining_data.drop(val_data.index)  # 20% for testing

        return train_data, val_data, test_data

    def build_model(self):
        """
        Builds the model architecture and compiles it.
        """
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=10))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def train_model(self, train_data, target_column, epochs=10, batch_size=32):
        """
        Trains the model on the training data.

        Args:
            train_data (pandas.DataFrame): Training data.
            target_column (str): Name of the target column in the training data.
            epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

        Returns:
            None
        """
    # Separate the features and the target column
        x_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate_model(self, val_data):
        """
        Evaluates the model on the validation data.

        Args:
            val_data (pandas.DataFrame): Validation data.

        Returns:
            tuple: A tuple containing the loss and accuracy of the model.
        """
        # Separate the features and the target column
        x_val = val_data.drop(columns=['target_column'])  # Replace 'target_column' with the actual name of the target column in your data
        y_val = val_data['target_column']  # Replace 'target_column' with the actual name of the target column in your data
        loss, accuracy = self.model.evaluate(x_val, y_val)
        return loss, accuracy

    def predict(self, new_data):
        """
        Makes predictions using the trained model.

        Args:
            new_data (numpy.ndarray): New data for prediction.

        Returns:
            numpy.ndarray: Predicted values.
        """
        prediction = self.model.predict(new_data)
        return prediction

    def save_model(self, model_file):
        """
        Saves the trained model to a file.

        Args:
            model_file (str): Path to the output model file.

        Returns:
            None
        """
        self.model.save(model_file)


# Example usage
if __name__ == '__main__':
    # Create an instance of the Recommender class
    # Omit columns_to_drop if not needed
    recommender = Recommender(columns_to_drop=['column1', 'column2'])  # Replace column names with actual ones to drop

    # Preprocess the data and customize column dropping
    train_data, val_data, test_data = recommender.preprocess_data('data.csv')

    # Build the model
    recommender.build_model()

    # Train the model
    recommender.train_model(train_data)

    # Evaluate the model on validation data
    loss, accuracy = recommender.evaluate_model(val_data)
    print(f'Validation Loss: {loss}, Accuracy: {accuracy}')

    # Use the trained model to make predictions
    new_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    prediction = recommender.predict(new_data)
    print(f'Prediction: {prediction}')

    # Save the model
    recommender.save_model('model.h5')
