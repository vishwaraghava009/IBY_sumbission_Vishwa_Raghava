# phoneme_to_morph.py
phoneme_to_morph_map = {
    'a': 'mouth_aaa_index', 'ɑ': 'mouth_aaa_index', 'ɐ': 'mouth_aaa_index', 'æ': 'mouth_aaa_index',
    'e': 'mouth_eee_index', 'ɛ': 'mouth_eee_index', 'ə': 'mouth_eee_index', 'ɜ': 'mouth_eee_index',
    'i': 'mouth_iii_index', 'ɪ': 'mouth_iii_index', 'ɨ': 'mouth_iii_index', 'ʉ': 'mouth_iii_index',
    'o': 'mouth_ooo_index', 'ɔ': 'mouth_ooo_index', 'oʊ': 'mouth_ooo_index', 'ɒ': 'mouth_ooo_index',
    'u': 'mouth_uuu_index', 'ʊ': 'mouth_uuu_index', 'ʌ': 'mouth_uuu_index', 'ɯ': 'mouth_uuu_index',
    'b': 'mouth_aaa_index', 'd': 'mouth_iii_index', 'f': 'mouth_eee_index', 'g': 'mouth_aaa_index',
    'h': 'mouth_aaa_index', 'j': 'mouth_iii_index', 'k': 'mouth_aaa_index', 'l': 'mouth_iii_index',
    'm': 'mouth_aaa_index', 'n': 'mouth_iii_index', 'p': 'mouth_aaa_index', 'r': 'mouth_ooo_index',
    's': 'mouth_eee_index', 'ʃ': 'mouth_uuu_index', 't': 'mouth_iii_index', 'v': 'mouth_eee_index',
    'w': 'mouth_uuu_index', 'z': 'mouth_iii_index', 'ʒ': 'mouth_uuu_index', 'ð': 'mouth_eee_index',
    'θ': 'mouth_uuu_index', 'ŋ': 'mouth_aaa_index', 'ç': 'mouth_eee_index', 'ʍ': 'mouth_uuu_index',
    'ɲ': 'mouth_iii_index', 'ɾ': 'mouth_ooo_index', 'ɡ': 'mouth_aaa_index', 'ɬ': 'mouth_eee_index',
    'ɮ': 'mouth_iii_index', 'ɹ': 'mouth_ooo_index', 'ɻ': 'mouth_ooo_index', 'ɽ': 'mouth_iii_index',
    'ɾ': 'mouth_ooo_index', 'ɭ': 'mouth_iii_index', 'ʂ': 'mouth_uuu_index', 'ʐ': 'mouth_iii_index',
    'ʈ': 'mouth_aaa_index', 'ɖ': 'mouth_iii_index', 'ɳ': 'mouth_aaa_index', 'ɲ': 'mouth_iii_index',
    'ʔ': 'mouth_aaa_index', 'ʡ': 'mouth_ooo_index', 'ɢ': 'mouth_aaa_index', 'ʢ': 'mouth_ooo_index',
    'ʘ': 'mouth_aaa_index', 'ǂ': 'mouth_ooo_index', 'ʝ': 'mouth_iii_index', 'ɣ': 'mouth_aaa_index',
    'χ': 'mouth_aaa_index', 'ʁ': 'mouth_ooo_index', 'ʕ': 'mouth_ooo_index', 'ħ': 'mouth_aaa_index',
    'ʜ': 'mouth_aaa_index', 'ɕ': 'mouth_iii_index', 'ʑ': 'mouth_iii_index', 'ɱ': 'mouth_aaa_index',
    'ɧ': 'mouth_aaa_index', 'ɳ': 'mouth_aaa_index', 'ɲ': 'mouth_iii_index', 'ɴ': 'mouth_aaa_index',
}

def phonemes_to_morph_indices(phonemes):
    morph_indices = []
    for phoneme in phonemes:
        morph_indices.append(phoneme_to_morph_map.get(phoneme, None))  #default is no morph if phoneme not found
    return morph_indices
