"""Unit test for process_data module.

Classes:
    Test_process_data
"""

from unittest import TestCase
import pandas as pd
import os

from data import process_data


class Test_process_data(TestCase):
    """Class used to test process_data module."""

    def setUp(self):
        """Constructs the test's environment ('fixture')."""

        self.messages_file = 'data/messages.csv'
        self.categories_file = 'data/categories.csv'

        if not os.path.exists(self.messages_file):
            print('No such file: {}'.format(self.messages_file))
            self.fail()

        if not os.path.exists(self.categories_file):
            print('No such file: {}'.format(self.categories_file))
            self.fail()

        self.messages = pd.read_csv(self.messages_file)
        self.categories = pd.read_csv(self.categories_file)

    def test_read_data(self):
        """Checks if number of rows between both datasets is equal to each other."""

        messages, categories = process_data.read_data(self.messages_file, self.categories_file)
        self.assertEqual(messages.shape[0], categories.shape[0])
