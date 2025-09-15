from typing import Dict, Optional

# Module-level alphabet configuration, set by setup_alphabet()
AA: Optional[str] = None
AA_TO_IDX: Optional[Dict[str, int]] = None
Q: Optional[int] = None


def setup_alphabet(seq_type: str):
    """Configure global alphabet for protein or RNA.

    Sets module-level variables: AA, AA_TO_IDX, Q
    """
    global AA, AA_TO_IDX, Q
    if seq_type == "protein":
        AA = "ACDEFGHIKLMNPQRSTVWY-"
        AA_TO_IDX = {a: i for i, a in enumerate(AA)}
    elif seq_type == "rna":
        AA = "ACGU-"
        AA_TO_IDX = {a: i for i, a in enumerate(AA)}
        # Map T/t to U channel
        AA_TO_IDX["T"] = AA_TO_IDX["U"]
        AA_TO_IDX["t"] = AA_TO_IDX["U"]
    else:
        raise ValueError(f"Unsupported seq_type: {seq_type}")
    Q = len(AA)
