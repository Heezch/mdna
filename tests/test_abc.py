import mdna as mdna

def test_make():
    n_bp = 10
    dna = mdna.make(n_bp=n_bp)
    assert len(dna.sequence) == n_bp