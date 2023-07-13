import unittest
import pandas as pd
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import os

class TestRecommendationSystem(unittest.TestCase):
    def setUp(self):
        # Load the data
        self.data = pd.read_csv('data.csv')

        # Preprocess the data
        self.data = self.data.drop(columns=['unnecessary_column'])
        self.data = self.data.fillna(0)

        # Split the data into training and testing sets
        self.train_data = self.data.sample(frac=0.8, random_state=0)
        self.test_data = self.data.drop(self.train_data.index)

        # Define the model architecture and compile it
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_dim=10))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model on the training set
        self.model.fit(self.train_data, epochs=10, batch_size=32)

        # Evaluate the model on the testing set
        self.loss, self.accuracy = self.model.evaluate(self.test_data)

        # Save the model for future use
        self.model.save('model.h5')

    def test_data_loaded(self):
        self.assertIsNotNone(self.data)

    def test_data_preprocessed(self):
        self.assertEqual(len(self.data.columns), 10)

    def test_data_split(self):
        self.assertEqual(len(self.train_data), 80)
        self.assertEqual(len(self.test_data), 20)

    def test_model_trained(self):
        self.assertGreater(self.accuracy, 0.5)

    def test_model_saved(self):
        self.assertTrue(os.path.exists('model.h5'))

    def test_model_loaded(self):
        loaded_model = load_model('model.h5')
        self.assertIsNotNone(loaded_model)

if __name__ == '__main__':
    unittest.main()
