from hprag.types import Claim, Verdict
from hprag.verification import EnsembleVerifier, LexicalVerifier, decompose

SRC = [
    "Halcyon was commissioned on 14 March 2041 by the Meridian Space Consortium.",
    "Smoking and open flames are never permitted anywhere on the station.",
    "Power output from the arrays increases in sunlight.",
]


def check(text, cited=None):
    c = Claim(text=text, cited=cited or [])
    LexicalVerifier().verify([c], SRC, threshold=0.55)
    return c


def test_supported_claim_with_valid_citation():
    c = check("Halcyon was commissioned in 2041.", cited=[1])
    assert c.verdict == Verdict.SUPPORTED
    assert c.evidence == [1] and c.citation_valid


def test_wrong_citation_is_flagged():
    c = check("Halcyon was commissioned in 2041.", cited=[2])
    assert c.verdict == Verdict.SUPPORTED and not c.citation_valid


def test_number_drift_is_contradiction():
    assert check("Halcyon was commissioned on 14 March 2039.").verdict == Verdict.CONTRADICTED


def test_negation_flip_is_contradiction():
    assert check("Open flames are permitted anywhere on the station.").verdict == Verdict.CONTRADICTED


def test_antonym_is_contradiction():
    assert check("Power output from the arrays decreases in sunlight.").verdict == Verdict.CONTRADICTED


def test_invented_fact_is_unsupported():
    assert check("The station has a nuclear reactor.").verdict == Verdict.UNSUPPORTED


def test_decompose_splits_claims_and_hedges():
    claims, hedges = decompose(
        "Halcyon launched in 2041 [1]. It has 9 crew [2]. The sources do not mention the commander."
    )
    assert [c.cited for c in claims] == [[1], [2]]
    assert len(hedges) == 1


def test_ensemble_is_conservative():
    class AlwaysYes:
        name = "yes"

        def verify(self, claims, sources, threshold):
            for c in claims:
                c.verdict, c.score, c.evidence = Verdict.SUPPORTED, 1.0, [1]
            return claims

    c = Claim(text="The station has a nuclear reactor.")
    EnsembleVerifier([LexicalVerifier(), AlwaysYes()]).verify([c], SRC, 0.55)
    assert c.verdict != Verdict.SUPPORTED  # one sceptical member is enough
