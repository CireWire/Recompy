# Recompy

Recompy is a Python library for building and training neural networks for recommender systems. The library is built on top of TensorFlow and Keras, providing a high-level API that simplifies the process of building, training, and evaluating recommender systems.

## Features

- Data preprocessing: Handle missing values and remove unnecessary columns.
- Model building: Construct neural network models for recommender systems.
- Training and evaluation: Train models on the provided data and evaluate their performance.
- Prediction: Make predictions using the trained models.
- Model saving: Save trained models for future use.

## Installation

You can install Recompy using pip:

```bash
pip install recompy
```
or 

```bash
pip install https://github.com/CireWire/Recompy/
```


## Usage

```python
import recompy

# Create an instance of the Recommender class
recommender = recompy.Recommender(columns_to_drop=['column1', 'column2'])

# Preprocess the data
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
```

For more details and advanced usage, please refer to the wiki.

## Docker

Recompy can also be run inside a Docker container. Docker provides an easy and consistent way to package and distribute your library as a containerized application.

To run Recompy using Docker, follow these steps:

1. Install Docker: Make sure you have Docker installed on your system. You can download and install Docker from the official Docker website ([https://www.docker.com](https://www.docker.com)).

2. Build the Docker image: Open a terminal, navigate to the project's root directory (where the Dockerfile is located), and run the following command to build the Docker image:

   ```bash
   docker build -t recompy .
    ```

This command will build the Docker image based on the instructions in the Dockerfile and tag it as recompy.

Run the Docker container: Once the image is built, you can run the Docker container using the following command:

  ```bash
  docker run recompy
  ```
This will start the container and execute the main Python script inside the container.

By running Recompy in a Docker container, you ensure that the library and its dependencies are isolated and can be easily deployed across different environments without worrying about compatibility issues.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This is project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) license.
