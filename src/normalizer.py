"""
Text normalization module for document classification.
"""
import re
from typing import Optional


class TextNormalizer:
    """Normalize text for document classification."""

    # Pattern to match month names (French + English)
    MONTHS_PATTERN = re.compile(
        r"\b(?:janv(?:ier)?|fev(?:rier)?|f[eé]vr(?:ier)?|mars|avr(?:il)?|mai|"
        r"juin|juil(?:let)?|ao[uû]t|sept(?:embre)?|oct(?:obre)?|nov(?:embre)?|"
        r"d[eé]c(?:embre)?|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\b",
        re.IGNORECASE,
    )

    # Pattern to match "part XXX" references
    PART_PATTERN = re.compile(r"\bpart\s+\d+\b", re.IGNORECASE)

    def __init__(self):
        """Initialize the text normalizer."""
        pass

    def normalize(self, text: Optional[str]) -> str:
        """
        Apply all normalization rules to the text.

        Rules:
        1. Convert to lowercase
        2. Replace newlines with spaces
        3. Remove month names (French + English)
        4. Preserve "part XXX" references (e.g., "part 145")
        5. Remove all other numbers
        6. Remove isolated special characters (preserve s/n, p/n patterns)
        7. Normalize whitespace

        Args:
            text: Raw text to normalize

        Returns:
            Normalized text string
        """
        if not isinstance(text, str) or not text:
            return ""

        # 1. Convert to lowercase
        text = text.lower()

        # 2. Replace newlines and carriage returns with spaces
        text = text.replace("\n", " ").replace("\r", " ")

        # 3. Remove month names
        text = self.MONTHS_PATTERN.sub("", text)

        # 4. Protect "part XXX" references by replacing temporarily
        part_references = []

        def protect_part(match):
            part_references.append(match.group(0))
            return f"__PART_{len(part_references) - 1}__"

        text = self.PART_PATTERN.sub(protect_part, text)

        # 5. Remove all numbers
        text = re.sub(r"\d+", "", text)

        # 6. Restore "part XXX" references
        for i, part_ref in enumerate(part_references):
            text = text.replace(f"__PART_{i}__", part_ref)

        # 7. Remove isolated special characters (preserve s/n, p/n, etc.)
        text = re.sub(r"\s+[^a-z0-9\s]+\s+", " ", text)

        # 8. Remove special characters at start/end
        text = re.sub(r"^[^a-z0-9\s]+", "", text)
        text = re.sub(r"[^a-z0-9\s]+$", "", text)

        # 9. Replace multiple consecutive special characters with one
        text = re.sub(r"([^a-z0-9\s])\1+", r"\1", text)

        # 10. Remove parentheses
        text = text.replace("(", " ").replace(")", " ")

        # 11. Normalize multiple spaces to single space
        text = re.sub(r"\s+", " ", text)

        # 12. Strip leading/trailing whitespace
        text = text.strip()

        return text
