#!/usr/bin/env python3
"""Compatibility wrapper. Prefer: urban-vlm ablate --logits ... --predictions ... --output ..."""
from urban_vlm.aggregation import main
if __name__ == "__main__":
    main()
