"""Front-end tests. Run with the venv:  python -m unittest discover -s tests"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "urdu"))

from tiny_voca import text  # noqa: E402
import frontend  # noqa: E402

frontend.install(text)


class FrontendTest(unittest.TestCase):
    def symbols(self, s):
        return [x for x in text.phonemize(s).symbols if x not in text.SPECIALS]

    def test_dataset_example_has_no_unknowns(self):
        ipa = "ɟˈʌɡˌa hɛ ɔr mˌariˈam keː bˈeːʈeː keː lˈiːeː nˈaːpnˌaː hɛ mˈẽ"
        p = text.phonemize("ipa:" + ipa)
        self.assertEqual(p.unknown, ())
        self.assertTrue(all(s in text.SYMBOLS for s in p.symbols))
        self.assertIn("dʒ", p.symbols)

    def test_every_ipa_char_in_audit_maps_into_inventory(self):
        chars = "aerkhntimˌəolsʌɪbdʊɟjɛpzʃqfʋɔuɡxcʂwɣʈʐɖɐɾʔðɹ"
        for c in chars:
            p = text.phonemize("ipa:" + c + "a")
            self.assertEqual(p.unknown, (), c)

    def test_aspiration_and_retroflex(self):
        self.assertEqual(self.symbols("ipa:kʰaː"), ["k", "h", "ɑː"])
        self.assertEqual(self.symbols("ipa:ʈeː"), ["t", "eɪ"])
        self.assertEqual(self.symbols("ipa:mẽ"), ["m", "eɪ", "n"])

    def test_roman(self):
        self.assertEqual(self.symbols("hai"), ["h", "ɛ"])
        self.assertEqual(self.symbols("aap"), ["ɑː", "p"])
        self.assertEqual(self.symbols("khana"), ["k", "h", "ɐ", "n", "ɑː"])
        self.assertEqual(self.symbols("acha"), ["ɐ", "tʃ", "ɑː"])

    def test_roman_sentence_structure(self):
        p = text.phonemize("Aap kaise hain?")
        self.assertEqual(p.symbols[0], text.BOS)
        self.assertEqual(p.symbols[-2:], ("?", text.EOS))
        self.assertIn(text.SP, p.symbols)
        self.assertEqual(p.unknown, ())

    def test_roman_never_unknown(self):
        for s in ("kya haal hai", "mujhe chai pasand hai", "shukriya, bohat shukriya!", "ghar jaldi aana"):
            self.assertEqual(text.phonemize(s).unknown, (), s)

    def test_nothing_to_pronounce(self):
        with self.assertRaises(text.NothingToPronounce):
            text.phonemize("123 ???")

    def test_fingerprint_is_urdu(self):
        self.assertEqual(text.frontend_identity()["phonemizer"], "tiny-urdu")


if __name__ == "__main__":
    unittest.main()
