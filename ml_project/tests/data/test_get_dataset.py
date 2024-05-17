import unittest

from ml_project.data import split_train_val_data, read_dataset
from ml_project.enities.split_params import SplittingParams

class TestGetDataset(unittest.TestCase):

    def test_read_dataset(self):
        df = read_dataset('ml_project/tests/test_data.csv')
        self.assertEqual(len(df), 1000)

    def test_split_data(self):
        df = read_dataset('ml_project/tests/test_data.csv')
        splitting_params = SplittingParams(random_state=42, val_size=0.3)
        train_df, test_df = split_train_val_data(df, splitting_params)
        self.assertEqual(len(train_df), 700)
        self.assertEqual(len(test_df), 300)


if __name__ == '__main__':
    unittest.main()
