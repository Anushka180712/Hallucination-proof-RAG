from hprag.text import (
    content_terms,
    extract_citations,
    has_negation,
    numbers,
    query_terms,
    split_sentences,
    stem,
)


def test_stem_conservative():
    assert stem("launched") == stem("launches") == "launch"
    assert stem("batteries") == "battery"
    assert stem("gas") == "gas"  # too short to strip


def test_split_sentences_keeps_abbreviations():
    s = split_sentences("The lead is Dr. Imani Okafor. She joined in 2043.")
    assert s == ["The lead is Dr. Imani Okafor.", "She joined in 2043."]


def test_split_sentences_separates_headings():
    assert split_sentences("# Title\nBody sentence here.") == ["# Title", "Body sentence here."]


def test_extract_citations():
    text, nums = extract_citations("Halcyon has 9 crew [1, 3]. ")
    assert text == "Halcyon has 9 crew."
    assert nums == [1, 3]


def test_numbers_normalised():
    assert numbers("It carries 3,200 kg and recovers 96%.") == {"3200", "96%"}


def test_negation_and_query_terms():
    assert has_negation("Flames are never permitted")
    assert not has_negation("Flames are permitted")
    assert "many" not in query_terms("How many crew members?")
    assert "crew" in content_terms("How many crew members?")


def test_irregular_verbs_share_a_stem():
    assert stem("began") == stem("begin") == "begin"
    assert stem("led") == "lead"
