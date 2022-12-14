import json
from typing import Dict, List
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
import executor
import stanza

import jmespath

import choppa
from choppa.srx_parser import SrxDocument
from choppa.iterators import SrxTextIterator
import re
import parse_wikidata
import pandas as pd
import wn

ACUTE = chr(0x301)
GRAVE = chr(0x300)
ruleset = Path(choppa.__file__).parent / "data/srx/languagetool_segment.srx"
SRX_2_XSD = Path(choppa.__file__).parent / "data/xsd/srx20.xsd"
document = SrxDocument(ruleset=ruleset, validate_ruleset=SRX_2_XSD)
PATTERN = re.compile("\)\s—")
PREFIX = "omw-en31"

nlp = stanza.Pipeline(lang="uk", processors="tokenize,mwt,pos,lemma", verbose=False)

COUNTER = 0


def remove_accents(s: str) -> str:
    return s.replace(ACUTE, "").replace(GRAVE, "")


# def lemmatize(s: str) -> List[str]:
#     doc = nlp(s)
#     return [word.lemma for sent in doc.sentences for word in sent.words]


def extract_from_page(word: str) -> Dict:
    try:
        s = executor.execute("node", "./wtf", word, capture=True)
        page = json.loads(s)
    except ValueError:
        page = {"json": dict()}
    merged_sections = defaultdict(list)
    for info_sections in jmespath.search("json.sections[].infoboxes[]", page) or {}:
        for k, v in info_sections.items():
            if "text" in v:
                merged_sections[k].append(v["text"])

    #     sentences: List[str] = []
    #     for s in page["plaintext"].splitlines():
    #         if not s.strip():
    #             continue

    #         for text in SrxTextIterator(document, "uk_one", s.strip(), max_lookbehind_construct_length=1024 * 10):
    #             if not text:
    #                 continue
    #             sentences.append(text)

    #     title = remove_accents(page["json"]["title"]).strip().lower()
    #     filtered_sentences: List[str] = []

    #     for sent in tqdm(sentences):
    #         if title in lemmatize(sent):
    #             filtered_sentences.append(sent)

    #     page["infoboxes"] = merged_sections
    #     page["examples"] = filtered_sentences
    if jmespath.search("json.sections[0].paragraphs[].sentences[0].text", page):
        gloss = jmespath.search("json.sections[0].paragraphs[].sentences[0].text", page)[0]
    else:
        gloss = ""
    if "—" in gloss:
        if re.search(PATTERN, gloss):
            _, gloss = re.split(PATTERN, gloss, 1)
        else:
            _, gloss = gloss.split("—", 1)
    page["gloss"] = gloss.strip()
    return page


if __name__ == "__main__":
    print(json.dumps(extract_from_page("Буряк"), indent=4, ensure_ascii=False))
