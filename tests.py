import unittest
import gpt_helpers

solver = gpt_helpers.LetterStringAnalogySolver()


class TestGPTHelperMethods(unittest.TestCase):

    def test_generate_prompt(self):
        prompt_data = [
            ["fff", "as", "ffs", "asdf"],
            ["abcd", "abce", "qwer", ""]
        ]
        self.assertEqual(
            "Q: if f f f changes to a s , what does f f s change to?\nA: a s d f\nQ: if a b c d changes to a b c e , what does q w e r change to?\nA: ",
            solver.generate_prompt(prompt_data)
        )


if __name__ == '__main__':
    unittest.main()
