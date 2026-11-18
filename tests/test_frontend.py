"""Front-end tests:  .venv/bin/python -m unittest discover -s tests"""
import glob, json, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "urdu"))
import frontend  # noqa: E402
frontend.install()
from tiny_voca import text  # noqa: E402

AUDIT_IPA = [  # real rows from the UrduTTS audit
    "ɟˈʌɡˌa hɛ ɔr mˌariˈam keː bˈeːʈeː keː lˈiːeː nˈaːpnˌaː hɛ",
    "ke dˌərəhəqˈiːqət kʰˈeːncənˌeː ʋˈaːlaː ˈeːk hˈiː hɛ lˈeːkɪn woː kˈʌi hˈoː kar naʐˈar ˈaːtaː hɛ",
    "ɪs mˈʌhəl keː ɡˈʌrd aːbˈaːd kjˈaː lˈoːɡõː neː ɟˈʌb dˈeːkʰaː ke bˌaːdʃˈaːh keːlˈoːkʰrˌiː",
    "xˌʊdaːˈoːn paːlˈaːneː pˈʌr dʰər.ˈaːl dˈiːaː hɛ tˈaːkeːh lˈoːɡõː keː amˈaːl dəkʰˈaːteː",
    "ke woː ɟˈʌnɡẽː lˈʌr.tˌiː hẽ ɪs keː nətˈiːɟeː mˈẽ mˈɪlneː ʋˈaːleː nˈʌfaː seː fˈaɪda ʊʈʰˈaːneː ʋˈaːleː bʰˈiː",
    "aɡˈar tˈoː mˈãː sˈaʋaː ˈallaːh sˈʌb seː fˈaːrɪɣ hɛ ˌamaːnˈeː hˈʌq mˈẽ hɛ ɟˈʌb tˈʌk",
]


class FrontendTest(unittest.TestCase):
    def phones(self, s):
        return [x for x in text.phonemize(s).symbols if x in text.PHONE_SET or x == text.UNK]

    def test_english_ids_unchanged(self):
        self.assertEqual(len(text.SYMBOLS), 76 + len(frontend.URDU_PHONES))
        self.assertEqual(text.SYMBOL_TO_ID["ɑː"], 53)

    def test_dataset_rows_have_no_unknowns(self):
        for ipa in AUDIT_IPA:
            self.assertEqual(text.phonemize("ipa:" + ipa).unknown, (), ipa)

    def test_native_phones_kept(self):
        self.assertEqual(self.phones("ipa:kʰaː"), ["kʰ", "aː"])
        self.assertEqual(self.phones("ipa:ʈeː"), ["ʈ", "eː"])
        self.assertEqual(self.phones("ipa:mˈẽ"), ["m", "ẽ"])
        self.assertEqual(self.phones("ipa:lˈoːɡõː"), ["l", "oː", "ɡ", "õ"])
        self.assertEqual(self.phones("ipa:ɽʰ"), ["ɽ", "h"])

    def test_roman(self):
        self.assertEqual(self.phones("hain"), ["h", "ẽ"])
        self.assertEqual(self.phones("aap"), ["aː", "p"])
        self.assertEqual(self.phones("khana"), ["kʰ", "a", "n", "aː"])
        self.assertEqual(self.phones("achha"), ["a", "cʰ", "aː"])
        self.assertEqual(self.phones("baRa"), ["b", "a", "ɽ", "aː"])
        self.assertEqual(self.phones("Tum"), ["t", "ʊ", "m"])          # sentence case is not retroflex

    def test_roman_sentences(self):
        p = text.phonemize("Aap kaise hain?")
        self.assertEqual(p.symbols[-2:], ("?", text.EOS))
        for s in ("kya haal hai", "mujhe chai pasand hai", "shukriya, bohat shukriya!", "ghar jaldi aana"):
            self.assertEqual(text.phonemize(s).unknown, (), s)

    def test_every_phone_is_an_aligner_label(self):
        found = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft/snapshots/*/vocab.json"))
        if not found:
            self.skipTest("aligner not downloaded")
        vocab = json.load(open(found[0], encoding="utf-8"))
        self.assertEqual([p for p in text.PHONES if p not in vocab], [])

    def test_config_and_model_follow_inventory(self):
        from tiny_voca.config import Config
        from tiny_voca import model
        self.assertEqual(Config().vocab_size, len(text.SYMBOLS))
        self.assertIn(text.SYMBOL_TO_ID["kʰ"], model.PHONE_IDS)
        self.assertEqual(text.frontend_identity()["phonemizer"], "tiny-urdu")


if __name__ == "__main__":
    unittest.main()
