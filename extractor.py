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

ACUTE = chr(0x301)
GRAVE = chr(0x300)
ruleset = Path(choppa.__file__).parent / "data/srx/languagetool_segment.srx"
SRX_2_XSD = Path(choppa.__file__).parent / "data/xsd/srx20.xsd"
document = SrxDocument(ruleset=ruleset, validate_ruleset=SRX_2_XSD)
PATTERN = re.compile("\)\s—")

nlp = stanza.Pipeline(lang="uk", processors="tokenize,mwt,pos,lemma", verbose=False)


# Todo: * data structure and traversal

def remove_accents(s: str) -> str:
    return s.replace(ACUTE, "").replace(GRAVE, "")


def lemmatize(s: str) -> List[str]:
    doc = nlp(s)
    return [word.lemma for sent in doc.sentences for word in sent.words]


def extract_from_page(word: str) -> Dict:
    s = executor.execute("node", "./wtf", word, capture=True)
    page = json.loads(s)

    merged_sections = defaultdict(list)
    for info_sections in jmespath.search("json.sections[].infoboxes[]", page) or {}:
        for k, v in info_sections.items():
            if "text" in v:
                merged_sections[k].append(v["text"])

    sentences: List[str] = []
    for s in page["plaintext"].splitlines():
        if not s.strip():
            continue

        for text in SrxTextIterator(document, "uk_one", s.strip(), max_lookbehind_construct_length=1024 * 10):
            if not text:
                continue
            sentences.append(text)

    title = remove_accents(page["json"]["title"]).strip().lower()
    filtered_sentences: List[str] = []

    for sent in tqdm(sentences):
        if title in lemmatize(sent):
            filtered_sentences.append(sent)

    page["infoboxes"] = merged_sections
    page["examples"] = filtered_sentences

    gloss = jmespath.search("json.sections[0].paragraphs[].sentences[0].text", page)[0] or ""
    if "—" in gloss:
        if re.search(PATTERN, gloss):
            _, gloss = re.split(PATTERN, gloss, 1)
        else:
            _, gloss = gloss.split("—", 1)
    page["gloss"] = gloss.strip()
    return page


if __name__ == "__main__":
    # print(json.dumps(extract_from_page("Буряк"), indent=4, ensure_ascii=False))

    # bfs to test
    visited = []
    queu = []
    X = extract_from_page("жук")["wikidata_id"]
    visited.append(X)
    queu.append(X)
    while queu:
        t = queu.pop(0)
        print(t)
        if not parse_wikidata.get_pwn_id(t):
            continue
        hypernyms = parse_wikidata.get_hypernyms(parse_wikidata.get_pwn_id(t))
        title_url = parse_wikidata.get_wiki_url(t)
        if not title_url:
            continue
        title = extract_from_page(title_url)['json']['title']
        print(title, ":", [extract_from_page(parse_wikidata.get_wiki_url(q))['json']['title'] for q in hypernyms if
                           q and parse_wikidata.get_wiki_url(q)])
        for x in hypernyms:
            if x and x not in visited:
                url = parse_wikidata.get_wiki_url(x)
                if not url:
                    continue
                new_page = extract_from_page(url)
                visited.append(x)
                queu.append(x)
