import unittest
import gpt_helpers

solver = gpt_helpers.LetterStringAnalogySolver()

class TestGPTHelperMethods(unittest.TestCase):

    def test_format_letter_string(self):
        self.assertEqual("a a a a", solver.format_letter_string("a a a a"))

    def test_generate_prompt(self):
        self.assertEqual("if a s d changes to d f d s , what does d f s change to?", solver.generate_prompt("asd", "dfds", "dfs"))


if __name__ == '__main__':
    unittest.main()