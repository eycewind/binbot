import unittest
import score_v2
import score_v3


class Tests(unittest.TestCase):
    def test_versioned_prompt_patch(self):
        original_prompt = score_v2.PROMPT
        original_version = score_v2.PROMPT_VERSION
        try:
            prompt = score_v3.configure()
            self.assertEqual(score_v2.PROMPT_VERSION, 'pilot_v3')
            self.assertIn('ETF flows are fund flows, not a BTC dollar-price quote', prompt)
            self.assertIn('label mixed and cite evidence for both', prompt)
            self.assertNotEqual(prompt, original_prompt)
        finally:
            score_v2.PROMPT = original_prompt
            score_v2.PROMPT_VERSION = original_version


if __name__ == '__main__':
    unittest.main()
