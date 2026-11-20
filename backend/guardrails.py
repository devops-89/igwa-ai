"""Scope guardrail and output sanity checks for backend services."""
import re
from collections import Counter

GUARDRAIL_REFUSAL_MESSAGE = (
    "यह प्रणाली केवल **भारतीय आपराधिक विधि (BNS 2023, BNSS 2023, BSB 2023), पुलिस अन्वेषण/विवेचना प्रक्रिया, साक्ष्य संकलन, तथा केस डायरी (Case Diary / पर्चा दैनिकी) प्रविष्टि** से संबंधित प्रश्नों के लिए अधिकृत है।\n\n"
    "❌ **अप्रासंगिक, सामान्य ज्ञान अथवा गैर-विवेचनात्मक प्रश्नों के उत्तर देना इस प्रणाली के कार्यक्षेत्र से बाहर है।**\n\n"
    "📌 कृपया केवल किसी आपराधिक घटना, पुलिस जांच प्रक्रिया, विधिक धारा, जब्ती/गिरफ्तारी अथवा केस डायरी प्रारूप से संबंधित प्रश्न ही पूछें।"
)

_IRRELEVANT = [
    re.compile(p) for p in (
        r"^(hi|hello|hey|greetings|how are you|who are you|kya haal hai|kaise ho)\b",
        r"\b(recipe|cook|bake|biryani|tea|coffee|maggi|pizza|burger|khana|sabzi)\b",
        r"\b(python|javascript|java|c\+\+|html|css|sql|function|react|angular|code|debug|script|compile)\b",
        r"\b(poem|poetry|shayari|joke|chutkula|movie|song|cinema|cricket|ipl|football|actor|actress|bollywood)\b",
        r"\b(capital of|weather in|temperature in|president of|prime minister of|who won the match)\b",
        r"\b(solve this equation|differentiate|integrate|calculate \d+)\b",
        # arithmetic only; '/' and '-' are excluded so FIR/case numbers and dates (12/2026) don't match
        r"(?<![\d/\-])\d+\s*[+*]\s*\d+(?![\d/\-])",
    )
]

_LEGAL_ASCII_TERMS = [
    "police", "investigation", "fir", "bns", "bnss", "bsb", "crpc", "arrest", "bail",
    "seizure", "case diary", "diary", "crime", "theft", "murder", "robbery", "cyber", "weapon",
    "assault", "harassment", "ndps", "pocso",
    # added: common investigation vocabulary (e.g. "Section 103 of the penal code" contains "code")
    "penal code", "section", "court", "accused", "complainant", "victim", "witness", "evidence",
    "chargesheet", "charge sheet", "panchnama",
]
# Word-boundary match (so "fir" no longer fires on "first"/"confirm"), tolerating simple suffixes.
_LEGAL_ASCII = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _LEGAL_ASCII_TERMS) + r")(?:s|es|ed|ing)?\b"
)
# Devanagari has no \b semantics, so substring matching is kept.
_LEGAL_DEVANAGARI = (
    "पुलिस", "विवेचना", "अन्वेषण", "केस डायरी", "पर्चा", "अपराध", "धारा", "गिरफ्तारी",
    "जमानत", "फर्द", "बरामदगी", "साक्ष्य", "गवाह", "चोरी", "हत्या", "लूट",
)


def is_police_investigation_query(text: str) -> bool:
    """False only if the text looks off-topic AND carries no legal/investigation vocabulary."""
    text = text.lower().strip()
    if not text:
        return False
    if not any(p.search(text) for p in _IRRELEVANT):
        return True
    return bool(_LEGAL_ASCII.search(text) or any(t in text for t in _LEGAL_DEVANAGARI))


def is_degenerate_output(response_text: str, repeat_threshold: int = 3, window: int = 8) -> bool:
    """Detects looping output: any `window`-word phrase repeated more than `repeat_threshold` times."""
    words = response_text.split()
    if len(words) < 20:
        return False
    counts = Counter(" ".join(words[i:i + window]) for i in range(len(words) - window + 1))
    return counts.most_common(1)[0][1] > repeat_threshold
