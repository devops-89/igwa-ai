import datetime
import threading
import sys
import time
import logging

# Suppress httpx INFO logs to keep the terminal clean
logging.getLogger("httpx").setLevel(logging.WARNING)

from rag import load_and_index_documents, llm

from llama_index.core import PromptTemplate
from translation_utils import (
    contains_hindi_script,
    is_hinglish_query,
    normalize_legal_hindi,
    translate_response_to_hindi,
)
from response_format import structure_response


def expand_scenario_for_law_query(user_input: str) -> str:
    """Rewrite natural-language scenario to push retrieval toward exact legal terms, definitions, and punishments across all Indian Penal Laws (BNS 2023 & IPC)."""
    text = user_input.lower()
    expansions = []

    patterns = {
        # Missing persons, kidnapping and abduction
        "missing|failed to return|gone missing|disappeared|not return|vanished|whereabouts unknown|kidnap|abduct|गुमशुदा|लापता|लौटकर नहीं आया|लौटकर नहीं आई|अपहरण|व्यपहरण|बहला-फुसलाकर ले जाना|अपने साथ ले गया|अपने साथ ले गई|स्कूल के बाहर से ले गया|स्कूल के बाहर से ले गई": [
            "missing person missing child kidnapping abduction wrongful confinement trafficking BNS 137 138 140 143 IPC 359 361 362 363 365 368 BNSS missing person report FIR search procedure",
        ],
        # Cyber offences and communications misuse
        "cyber|social media|online account|message|profile|whatsapp|email|it act|digital evidence|cyberbullying|online threat|साइबर|साइबर अपराध|सोशल मीडिया|ऑनलाइन खाता|संदेश|धमकी|इंटरनेट पर|फर्जी प्रोफाइल|व्हाट्सऐप|ईमेल": [
            "cyber crime information technology act identity theft cheating by personation privacy violation obscene electronic material stalking criminal intimidation BNS 77 78 79 318 319 351 IT Act 66 66C 66D 66E 67 electronic evidence",
        ],
        # Bank and payment fraud; hints support investigation, not an assumed offence
        "bank|unauthorized transaction|unauthorised transaction|bank account|account debit|withdrawn|withdrawal|upi|atm|debit card|credit card|net banking|phishing|otp|बैंक|बैंक खाता|खाते से राशि|अनधिकृत लेनदेन|अनधिकृत निकासी|राशि निकाली|रुपये निकाले|खाता डेबिट|यूपीआई|एटीएम|ओटीपी|फिशिंग|धोखाधड़ी": [
            "bank unauthorized transaction UPI ATM card net banking phishing OTP identity theft cheating electronic record IT Act 66C 66D BNS 318 319 beneficiary account transaction reference authentication consent bank records",
        ],
        # Trespass, house-breaking and civil possession disputes
        "illegal occupation|occupy|occupied|entered house|entered the house|entered the home|unauthorized entry|trespass|recovery of property|possession|dispossession|burglary|break-in|attempted burglary|housebreaking|locked office|कब्जा|अवैध कब्जा|जमीन|भूमि|सीमा विवाद|जमीन की सीमा|भूखंड|निर्माण कार्य|घर में प्रवेश|घर में घुस|मकान में प्रवेश|मकान में घुस|अनधिकृत प्रवेश|अतिचार|गृह-अतिचार|संपत्ति से बेदखल": [
            "criminal trespass house trespass house breaking lurking house trespass BNS 329 330 331 332 333 IPC 441 442 445 448 452 453 Specific Relief Act sections 5 6 possession wrongful dispossession civil remedy boundary dispute",
        ],
        # Property theft, robbery, extortion and breach of trust
        "stolen|theft|vehicle stolen|goods stolen|steal|snatching|robbery|dacoity|extortion|valuables|cash and jewellery|cash and jewelry|cash and mobile|cbt|breach of trust|चोरी|चुराया|चुरा लिया|लूट|लूटपाट|डकैती|छीनना|छीनने का प्रयास|छीनने की कोशिश|झपटमारी|सामान लेकर फरार|पैसे और मोबाइल|मोबाइल फोन छीन|नकदी और आभूषण|नकदी तथा आभूषण|आभूषण लिए|उद्यापन|जबरन वसूली|गबन|न्यासभंग|वाहन चोरी|नकदी लेकर": [
            "theft snatching extortion robbery dacoity criminal misappropriation criminal breach of trust receiving stolen property BNS 303 304 308 309 310 314 316 317 IPC 378 379 383 390 391 392 395 405 406 411 definitions punishment classification",
        ],
        # Threats and demands for property
        "criminal intimidation|intimidation|threatened|threatening|threat|धमकी|धमकाया|धमकी दी|धमकाकर|डराया": [
            "criminal intimidation threat to person reputation property intent to cause alarm compel act BNS section 351 IPC 503 506 evidence call records messages",
        ],
        "demanded money|demand for money|demand for payment|ransom|money demand|पैसे की मांग|रुपये की मांग|रुपए की मांग|धन की मांग|लाख की मांग|पैसे नहीं देने पर|फिरौती|जबरन वसूली|उद्यापन": [
            "extortion putting a person in fear of injury to deliver property BNS section 308 IPC 383 384 criminal intimidation BNS 351 threat evidence payment demand",
        ],
        "wrongful restraint|रास्ता रोक|आगे जाने नहीं दिया|जाने से रोक": [
            "wrongful restraint BNS section 126 IPC 339 341 voluntarily obstructing a person from proceeding in a direction evidence movement liberty",
        ],
        # Assault, hurt and group liability
        "assault|beaten|hit|struck|attacked|injured|struggle|hurt|grievous hurt|public servant|मारपीट|मारपीट की|पीटा|पिटाई|हमला|हमला किया|चोट|घायल|घायल हुए|कहासुनी|झगड़ा|लड़ाई|हाथापाई|उपहति|घोर उपहति|धक्का|थप्पड़": [
            "voluntarily causing hurt grievous hurt criminal force assault BNS 115 116 117 118 121 131 132 IPC 319 320 321 323 325 351 common intention BNS 3(5) IPC 34 medical evidence cross complaint",
        ],
        # Cheating, forgery and impersonation
        "fraud|cheating|forged cheque|online scam|fake identity|loan|forgery|false document|impersonation|documents were used|used the documents|checkbook|check book|धोखाधड़ी|छल|ठगी|जालसाजी|कूटरचना|फर्जी दस्तावेज|जाली दस्तावेज|दस्तावेज नकली|दस्तावेज़ नकली|नकली थे|दस्तावेजों का इस्तेमाल|दस्तावेजों का उपयोग|दस्तावेजों का दुरुपयोग|दस्तावेज का इस्तेमाल|दस्तावेज का उपयोग|दस्तावेज का दुरुपयोग|चेकबुक|फर्जी पहचान|प्रतिरूपण|नकली चेक|ऋण": [
            "cheating fraud cheating by personation forgery false document use forged document criminal conspiracy BNS 61 318 319 335 336 337 338 340 IPC 120B 415 416 420 463 465 468 471 documentary electronic evidence",
        ],
        # Death, homicide and attempts (do not confuse completed death with attempt)
        "murder|dead|death|killed|assassinated|homicide|body found|attempt to murder|मृत|मृत्यु|मौत|हत्या|हत्या का प्रयास|जान से मार|खून|शव|मृत पाया|मृत पाई": [
            "murder culpable homicide causing death attempt to murder BNS 100 101 103 105 109 IPC 299 300 302 304 307 postmortem forensic evidence death investigation",
        ],
        # Damage, arson and mischief
        "mischief|damage to property|destruction|crops|vandalism|fire|explosion|तोड़फोड़|संपत्ति को नुकसान|नुकसान पहुंचाया|फसल नष्ट|आग लगाई|आग लग गई|जानबूझकर आग|आग के जानबूझकर लगाए|आग लगाए जाने|आगजनी|विस्फोट|नुकसान किया": [
            "mischief property damage destruction arson crop damage BNS 324 326 IPC 425 427 435 436 fire explosion valuation evidence",
        ],
        # Weapons and explosives
        "firearm|arms|illegal weapons|gun|pistol|revolver|ammunition|explosive|हथियार|अवैध हथियार|बंदूक|पिस्तौल|रिवॉल्वर|गोली|कारतूस|आग्नेयास्त्र|विस्फोटक": [
            "Arms Act illegal possession firearm ammunition use weapon BNS section 25 27 Arms Act explosive substances act seizure ballistic evidence licence",
        ],
        # Narcotics and prohibited substances
        "narcotic|drug|ganja|cocaine|heroin|opium|contraband|prohibited substance|नशीला पदार्थ|मादक पदार्थ|नारकोटिक्स|गांजा|चरस|कोकीन|हेरोइन|अफीम|प्रतिबंधित पदार्थ|मादक द्रव्य": [
            "NDPS Act narcotic drugs psychotropic substance possession sale trafficking sections 8 20 21 22 search seizure sampling chain of custody chemical analysis",
        ],
        # Conspiracy, abetment and joint participation
        "conspiracy|accomplice|pre-planned|joint liability|abetment|षड्यंत्र|साजिश|आपराधिक षड्यंत्र|दुष्प्रेरण|सह-अभियुक्त|साझा आशय|समान आशय|सामान्य आशय|सामान्य उद्देश्य|साथियों के साथ|दो साथियों|साथ मिलकर|तीनों आरोपी|तीनों ने|मिलकर अपराध": [
            "criminal conspiracy common intention common object abetment BNS 3(5) 45 46 61 189 190 IPC 34 107 109 120B evidence agreement participation",
        ],
        # Offences against women
        "rape|sexual assault|outrage modesty|stalking|voyeurism|dowry death|dowry harassment|domestic violence|cruelty to woman|acid attack|molest|molestation|eve teasing|बलात्कार|दुष्कर्म|यौन हमला|लैंगिक हमला|छेड़छाड़|यौन उत्पीड़न|पीछा करना|पीछा कर रहा|पीछा करता|पीछा किया|घूरना|दहेज उत्पीड़न|दहेज मृत्यु|घरेलू हिंसा|महिला के साथ क्रूरता|तेजाब हमला": [
            "BNS offences against women rape sexual assault sexual harassment voyeurism stalking dowry death cruelty husband acid attack 63 64 65 74 75 77 78 79 80 85 86 124 IPC 304B 326A 354 354A 354C 354D 375 376 498A POCSO Dowry Prohibition Act",
        ],
        # Child protection and child labour/trafficking
        "pocso|child sexual assault|child abuse|minor girl|minor boy|child kidnapping|child trafficking|child abandonment|child labor|minor child|children|पॉक्सो|बाल यौन शोषण|बच्चे का अपहरण|नाबालिग|नाबालिग लड़की|नाबालिग लड़का|बाल तस्करी|बाल श्रम|बाल विवाह|बालक": [
            "POCSO Act child sexual assault penetrative sexual assault sexual harassment child trafficking kidnapping BNS 137 140 143 146 trafficking Juvenile Justice Act Child Labour Act sections 3 4 5 6 7 8 9 10 11 12",
        ],
        # Public order, rioting and unlawful assembly
        "riot|rioting|unlawful assembly|affray|public order|mob|violent protest|दंगा|उपद्रव|अवैध जमाव|गैरकानूनी जमाव|भीड़ ने हमला|सार्वजनिक शांति|झड़प|सामूहिक हिंसा": [
            "unlawful assembly rioting affray public tranquillity common object BNS 189 190 191 194 IPC 141 146 147 148 149 public order evidence",
        ],
        # Obstruction, disobedience and public-servant offences
        "obstruct|obstruction|resist|disobey|public servant|government officer|सरकारी काम में बाधा|पुलिस से हाथापाई|लोक सेवक|सरकारी कर्मचारी|आदेश का उल्लंघन": [
            "obstructing public servant disobedience lawful order assault public servant BNS 132 221 223 IPC 186 188 353 lawful duty evidence",
        ],
        # False evidence, destruction of evidence and investigation obstruction
        "false evidence|perjury|fabricated evidence|destroy evidence|tamper evidence|conceal evidence|झूठी गवाही|मिथ्या साक्ष्य|साक्ष्य मिटाना|सबूत मिटाना|साक्ष्य से छेड़छाड़|दस्तावेज नष्ट|गवाह को धमकी": [
            "false evidence fabricating false evidence destruction of evidence screening offender threatening witness BNS 227 228 238 232 IPC 191 192 201 195A electronic documentary evidence",
        ],
        # Motor-vehicle and road incidents
        "road accident|motor accident|traffic accident|hit and run|drunk driving|rash driving|negligent driving|सड़क दुर्घटना|वाहन दुर्घटना|हिट एंड रन|टक्कर|लापरवाही से वाहन|तेज रफ्तार|नशे में गाड़ी|सड़क हादसा": [
            "motor vehicle road accident rash negligent driving BNS 281 106 125 Motor Vehicles Act 134 185 187 hit and run injury death medical mechanical inspection CCTV",
        ],
        # Food, environmental and animal offences
        "food adulteration|adulterated food|poisoning|environmental pollution|illegal mining|animal cruelty|cattle theft|food safety|खाद्य मिलावट|मिलावटी भोजन|जहर|प्रदूषण|अवैध खनन|पशु क्रूरता|पशुओं पर अत्याचार|गौ-तस्करी|खाद्य सुरक्षा": [
            "Food Safety and Standards Act adulteration poisoning environmental protection pollution illegal mining Prevention of Cruelty to Animals Act animal cruelty BNS public safety applicable special statute laboratory evidence",
        ],
        # Marriage and domestic offences
        "bigamy|marriage fraud|child marriage|domestic abuse|dowry|द्विविवाह|दूसरी शादी|बाल विवाह|घरेलू प्रताड़ना|वैवाहिक हिंसा|दहेज|स्त्रीधन|पति द्वारा क्रूरता": [
            "bigamy child marriage domestic violence dowry demand cruelty husband relatives BNS 82 85 86 Dowry Prohibition Act sections 2 4 8 Prohibition of Child Marriage Act Protection of Women from Domestic Violence Act",
        ],
    }

    for pattern, keywords in patterns.items():
        if any(key in text for key in pattern.split("|")):
            expansions.extend(keywords)

    # Retrieval hints are kept separate from the actual user-provided facts.
    expansions.append(
        "BNSS 2023 First Schedule classification of offences: cognizable or non-cognizable; "
        "bailable or non-bailable; classification for the exact BNS section and subsection"
    )
    return " ".join(dict.fromkeys(expansions))


def create_query_engines(index, similarity_top_k=6,top_p = 0.8 ,response_mode="compact"):
    """Create query engines with specified retrieval parameters for different languages"""
    
    _BASE = (
        "RETRIEVED LEGAL CONTEXT (use only to verify legal provisions; it is not a source of scenario facts):\n"
        "{context_str}\n\n"
        "USER-PROVIDED SCENARIO (the sole source of case facts; read every sentence carefully):\n"
        "{query_str}\n\n"
        "First extract the incidents explicitly stated in the scenario: preserve their order, actors, actions, and stated outcomes. Include only those incidents. Every factual claim in the answer must be traceable to the scenario; do not fill gaps with typical or assumed details. Retrieval terms and retrieved legal passages are never case facts.\n"
        "Do not add events, people, motives, injuries, weapons, threats, property loss, entry, or conduct unless the scenario states them. If a relevant detail is absent, say it is not stated; do not discuss unrelated hypothetical offences or incidents. Investigation steps may ask how to verify stated facts, but must not imply any unstated fact occurred. Ignore illegible OCR, broken text, headers, and unrelated passages.\n"
        "Identify every distinct offence plausibly supported by the stated facts, and connect each offence to the facts and required elements. For dowry allegations, separately assess BNS sections 85/86 (section 86 defines cruelty and is not a separate offence) and Dowry Prohibition Act section 4 only if demand was made directly or indirectly from the bride's/bridegroom's parents, relatives, or guardian; do not infer the recipient if unstated. For each offence, include applicable readable sections and subsections, including aggravated or attempt provisions only when facts support them. Do not count definitions, alternatives, or penalties as separate offences. Do not infer guilt, intent, or facts.\n"
        "Use only legal provisions and punishments clearly supported by relevant retrieved text. If a detail is missing or unreadable, mark it unverified instead of guessing. Report cognizability and bail status only when the exact schedule entry is available; otherwise mark them unverified.\n"
        "Distinguish completed offences from attempts. If the scenario says a person died, do not describe that death as an attempt to murder. Identify possible homicide offences only as legal possibilities supported by the stated facts, and do not conclude who caused the death or with what intent without evidence.\n"
        "Do not infer an offence solely from damaged property, blood, an argument, a phone call, or a person's presence or visit. These may be investigative leads, not proof of an offence or guilt. Do not invent unrelated allegations or carry over issue-specific cautions from other scenarios.\n"
        "It is valid to state that the facts do not establish a particular offence or section. Omit irrelevant sections and cautions entirely.\n"
        "Skip any text with '...........' blanks. Do not mention file names, file paths, or PDF names.\n\n"
        "INSTRUCTIONS:\n"
        "Write five complete, focused sections: facts; all distinct potential offences with count and supporting facts; legal provisions with exact Act/section/subsection and punishment for every offence; classification; and 3-5 specific investigation steps. In the facts section, explicitly state whether physical harm is alleged: Yes/No/Not stated, paired with हाँ/नहीं/उल्लेख नहीं. In classification, state the nature of each offence: cognizable or non-cognizable and bailable or non-bailable. Use the exact BNSS First Schedule or applicable special-law schedule row for each section, preserve conditional classifications, and show each status in both English and Hindi (Cognizable: ... / संज्ञेयता: ...; Bailable: ... / जमानतीय: ...). If a row is missing or unclear, say Unverified / सत्यापित नहीं for that status rather than guessing. Distinguish offences from definition sections and do not double-count definitions or alternative subsections. Do not omit a fact-supported offence or add an unsupported one.\n"
    )
    _auto    = _BASE + ("Detect the language of the scenario. Keep all prose in that language, with both languages used only for the required physical-harm and classification labels. "
        "If Hindi or Hinglish, write the answer prose in formal Hindi (Devanagari). "
        "If English, write the answer prose in English. Return exactly five complete, focused Markdown sections in the detected language: facts summary, offences/issues, legal provisions, classification, investigation steps.\nAnswer:\n")
    _hindi   = _BASE + ("IMPORTANT: Write your ENTIRE answer in formal Hindi (Devanagari script) "
        "as used by the Indian Government, Indian Police, and Indian Courts. "
        "Understand the Hindi scenario before applying law. Summarize its actual facts in clear, natural Hindi; do not replace them with a generic template. Explain each legal possibility in complete, plain Hindi sentences, not copied English fragments or literal word-for-word translations. Preserve every condition, section number, and punishment accurately. Use exactly these five headings, in order: '### तथ्यों का संक्षेप', '### अपराध और विधिक मुद्दे', '### लागू विधिक प्रावधान', '### वर्गीकरण', '### अन्वेषण के चरण'. Number only the investigation steps. Keep statute names and section numbers accurate. "
        "Do NOT write English prose, Hinglish, or transliterated English except for the required bilingual physical-harm and classification labels.\nAnswer:\n")
    _english = _BASE + ("Write the answer prose in English. Use Hindi only for the required bilingual physical-harm and classification labels. "
        "Return exactly five sections: '### Facts considered', '### Potential offences and issues', '### Applicable legal provisions', '### Classification', '### Investigation steps'.\nAnswer:\n")

    def _engine(tmpl_str):
        return index.as_query_engine(
            llm=llm,
            similarity_top_k=similarity_top_k,
            response_mode=response_mode,
            text_qa_template=PromptTemplate(tmpl_str),
        )

    return {
        "auto":    _engine(_auto),
        "hindi":   _engine(_hindi),
        "english": _engine(_english),
    }


def print_sources(response, max_chars=150):
    """Print retrieved source chunks so the officer can verify provenance"""
    if not getattr(response, "source_nodes", None):
        return
    print("\nSources:")
    for i, node in enumerate(response.source_nodes, 1):
        meta = node.node.metadata or {}
        file_name = meta.get("file_name", meta.get("file_path", "unknown"))
        snippet = node.node.get_content().strip().replace("\n", " ")[:max_chars]
        print(f"  [{i}] {file_name}")
        print(f"      {snippet}...")


def log_interaction(query, response, log_path="query_log.txt"):
    """Append query/response/sources to a plain-text audit log.

    NOTE: if this system handles sensitive case data, treat this log file's
    access and retention the same as any other case file, not as a throwaway
    debug log.
    """
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now().isoformat()}]\n")
            f.write(f"Query: {query}\n")
            f.write(f"Response: {response}\n")
            if getattr(response, "source_nodes", None):
                for i, node in enumerate(response.source_nodes, 1):
                    meta = node.node.metadata or {}
                    f.write(f"  Source [{i}]: {meta.get('file_name', 'unknown')}\n")
            f.write("-" * 50 + "\n")
    except Exception as e:
        print(f"(Warning: could not write to log — {e})")


def detect_degenerate_output(response_text, repeat_threshold=3):
    """Detect degenerate repetitive output that indicates model failure."""
    words = response_text.split()
    if len(words) < 20:
        return False, None
    phrase_len = 8
    phrase_counts = {}
    for i in range(len(words) - phrase_len + 1):
        phrase = " ".join(words[i:i + phrase_len])
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        if phrase_counts[phrase] > repeat_threshold:
            return True, phrase
    return False, None


def interactive_query_loop(index):
    """Run an interactive query loop with the user"""
    engines = create_query_engines(index)
    current_lang = "auto"

    print("\nIGWA Legal RAG System Ready")
    print("Describe a scenario to get legal analysis.")
    print("Type 'exit' or 'quit' to stop.")
    print("Type '/lang hindi', '/lang english', '/lang both', or '/lang auto' to switch languages.\n")

    while True:
        try:
            user_input = input("Describe the scenario: ").strip()
            if not user_input:
                continue
                
            lower_input = user_input.lower()
            if lower_input in ["exit", "quit"]:
                print("Exiting...")
                break
                
            if lower_input.startswith("/lang "):
                lang = lower_input.split(" ")[1]
                if lang in engines or lang == "both":
                    current_lang = lang
                    print(f"Language switched to: {lang.upper()}")
                else:
                    print(f"Unknown language: {lang}. Use auto, hindi, english, or both.")
                continue

            if current_lang == "auto":
                if contains_hindi_script(user_input):
                    response_lang = "hindi"
                elif is_hinglish_query(user_input):
                    response_lang = "both"
                else:
                    response_lang = "english"
            else:
                response_lang = current_lang

            print(f"\nAnalysing (Language: {response_lang.upper()})...\n")
            run_engine = engines["hindi" if response_lang == "hindi" else "english"]
            retrieval_hints = expand_scenario_for_law_query(user_input)
            from llama_index.core import QueryBundle

            retrieval_bundle = QueryBundle(
                query_str=user_input,
                custom_embedding_strs=[user_input, retrieval_hints] if retrieval_hints else [user_input],
            )
            response = run_engine.query(retrieval_bundle)
            response_text = str(response)

            if response_lang == "hindi":
                response_text = normalize_legal_hindi(response_text)

            if response_lang == "hindi":
                hindi_response = response_text
            elif response_lang == "both":
                print("Translating to Hindi...")
                hindi_response = translate_response_to_hindi(response_text)

            response_text, formatting_fallback = structure_response(
                response_text,
                user_input,
                "hindi" if response_lang == "hindi" else "english",
            )
            if response_lang == "hindi":
                hindi_response = response_text
            elif response_lang == "both":
                hindi_response, _ = structure_response(hindi_response, user_input, "hindi")

            # Guard against degenerate repetitive output
            is_degenerate, repeated_phrase = detect_degenerate_output(response_text)
            if is_degenerate or formatting_fallback:
                print("Warning: The generated answer could not be used as-is.")
                if formatting_fallback:
                    print("The model returned text that did not match the required format; showing a simplified fallback. Verify legal details against the source documents.")
                else:
                    print("Try rephrasing with more specific details.")
            if response_lang == "hindi":
                print(hindi_response)
            elif response_lang == "both":
                print("English response:\n")
                print(response_text)
                print("\nHindi response:\n")
                print(hindi_response)
            else:
                print(response_text)

            print_sources(response)
            log_interaction(user_input, response)
            print("\n" + "-" * 50 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")
            print("-" * 50)


# Main execution
if __name__ == "__main__":

    print("Starting RAG Pipeline...")

    try:
        # Load or create index
        index = load_and_index_documents()

        # Start interactive loop
        interactive_query_loop(index)

    except Exception as e:
        print(f"\nSystem Error: {str(e)}")
