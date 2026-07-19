"""
Neuro-Symbolic Rule Engine for CODE-ACCORD.
Filters predicted triples based on ontology constraints derived from training data.
"""

# Allowed (head_type, relation_type, tail_type) triples.
# Derived from training data statistics with count >= 1.
ALLOWED_COMBINATIONS = {
    # Comparison rules (Value is usually the tail)
    ("Property", "greater-equal", "Value"),
    ("Property", "less-equal", "Value"),
    ("Property", "equal", "Value"),
    ("Property", "greater", "Value"),
    ("Property", "less", "Value"),
    ("Object", "greater-equal", "Value"),
    ("Object", "less-equal", "Value"),
    ("Object", "greater", "Value"),
    ("Object", "less", "Value"),
    ("Object", "equal", "Value"),
    ("Value", "greater-equal", "Property"),
    ("Value", "greater-equal", "Object"),
    ("Value", "equal", "Object"),
    ("Value", "equal", "Property"),
    
    # Necessity / Selection rules
    ("Object", "necessity", "Quality"),
    ("Object", "selection", "Quality"),
    ("Object", "necessity", "Object"),
    ("Object", "selection", "Object"),
    ("Object", "necessity", "Property"),
    ("Object", "selection", "Property"),
    ("Property", "necessity", "Quality"),
    ("Property", "selection", "Quality"),
    ("Property", "necessity", "Object"),
    ("Property", "selection", "Object"),
    ("Property", "necessity", "Property"),
    ("Property", "selection", "Property"),
    ("Quality", "selection", "Object"),
    ("Quality", "selection", "Property"),
    ("Quality", "selection", "Quality"),
    ("Quality", "necessity", "Object"),
    ("Quality", "necessity", "Quality"),
    ("Value", "necessity", "Quality"),
    ("Value", "selection", "Property"),
    ("Value", "selection", "Quality"),
    ("Value", "selection", "Object"),

    # Structural rules (part-of)
    ("Object", "part-of", "Object"),
    ("Property", "part-of", "Object"),
    ("Object", "part-of", "Property"),
    ("Property", "part-of", "Property"),
    ("Value", "part-of", "Property"),
    ("Value", "part-of", "Object"),
    ("Object", "not-part-of", "Object"),
    ("Object", "not-part-of", "Property"),
    ("Property", "not-part-of", "Property"),
    ("Object", "part-of", "Quality"),
    ("Property", "part-of", "Quality"),
    ("Quality", "part-of", "Object"),
    ("Quality", "part-of", "Property"),
}

def is_valid_triple(head_type, relation, tail_type):
    """Check if a triple is allowed by the ontology rules."""
    # Normalize types to match our dictionary keys
    h = head_type.capitalize()
    t = tail_type.capitalize()
    r = relation.lower()
    
    # We allow "Unknown" to pass for now to avoid being too aggressive,
    # or we can block it. Let's block it for a "hard" rule.
    if h == "Unknown" or t == "Unknown":
        return False
        
    return (h, r, t) in ALLOWED_COMBINATIONS

def filter_triples(triples):
    """
    triples: list of dicts with {head, head_type, relation, tail, tail_type, confidence}
    """
    valid = []
    filtered_count = 0
    for tri in triples:
        if is_valid_triple(tri["head_type"], tri["relation"], tri["tail_type"]):
            valid.append(tri)
        else:
            filtered_count += 1
    return valid, filtered_count
