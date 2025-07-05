import unittest
from src.data_feed import DataFeed

class TestDataFeed(unittest.TestCase):

    def setUp(self):
        self.data_feed = DataFeed()

    def test_fetch_data_valid(self):
        result = self.data_feed.fetch_data("CBA.AX")
        self.assertIsNotNone(result)
        self.assertIn("CBA.AX", result)

    def test_fetch_data_invalid(self):
        result = self.data_feed.fetch_data("INVALID_SYMBOL")
        self.assertIsNone(result)

    def test_fetch_data_empty_symbol(self):
        result = self.data_feed.fetch_data("")
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()