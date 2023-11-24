import re
import json

import scrapy
import wn
import jmespath

from pwn_friends.items import PwnFriendsItem

WIKIDATA_QUERY = """PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?item ?itemLabel ?en_wiki ?uk_wiki ?pl_wiki ?ru_wiki WHERE {{
  ?item wdt:{prop_name} "{prop_value}" .

    OPTIONAL {{
      ?en_wiki schema:about ?item .
      ?en_wiki schema:inLanguage "en" .
      ?en_wiki schema:isPartOf <https://en.wikipedia.org/> .
    }}
    OPTIONAL {{
      ?uk_wiki schema:about ?item .
      ?uk_wiki schema:inLanguage "uk" .
      ?uk_wiki schema:isPartOf <https://uk.wikipedia.org/> .
    }}
    OPTIONAL {{
      ?pl_wiki schema:about ?item .
      ?pl_wiki schema:inLanguage "pl" .
      ?pl_wiki schema:isPartOf <https://pl.wikipedia.org/> .
    }}
    OPTIONAL {{
      ?ru_wiki schema:about ?item .
      ?ru_wiki schema:inLanguage "ru" .
      ?ru_wiki schema:isPartOf <https://ru.wikipedia.org/> .
    }}
}}"""

PWN_URL = "https://en-word.net/json/id/oewn-{pwn_id}"
LEXICON = "omw-en31"
WIKIDATA_URL = "https://query.wikidata.org/sparql"

PWN31_WD_PROP = "P8814"
ILI_WD_PROP = "P5063"


class PwnWikidataSpider(scrapy.Spider):
    name = "pwn_wikidata"
    allowed_domains = [
        "wikidata.org",
        "wordnet-rdf.princeton.edu",
        "query.wikidata.org",
    ]
    start_urls = ["http://wikidata.org/"]

    def start_requests(self):
        for wn_synset in wn.synsets(lexicon=LEXICON):
            yield scrapy.Request(
                PWN_URL.format(pwn_id=wn_synset.id.replace(f"{LEXICON}-", "")),
                callback=self.parse_pwn,
            )

    def parse_pwn(self, response):
        json_response = json.loads(response.body)
        for synset in json_response:
            if synset.get("ili"):
                yield PwnFriendsItem(
                    id_from=f"{LEXICON}-{synset['id']}",
                    id_to=f"ili-{synset['ili']}",
                    rel="pwn31_to_ili",
                )

                yield scrapy.FormRequest(
                    WIKIDATA_URL,
                    method="GET",
                    formdata={
                        "query": WIKIDATA_QUERY.format(
                            prop_name=ILI_WD_PROP, prop_value=synset["ili"]
                        )
                    },
                    headers={"Accept": "application/sparql-results+json"},
                    meta={
                        "id_from": f"ili-{synset['ili']}",
                        "rel_prefix": "ili",
                        "dont_cache": True,
                    },
                    callback=self.parse_wd_pwn,
                )

            yield scrapy.FormRequest(
                WIKIDATA_URL,
                method="GET",
                formdata={
                    "query": WIKIDATA_QUERY.format(
                        prop_name=PWN31_WD_PROP, prop_value=synset["id"]
                    )
                },
                headers={"Accept": "application/sparql-results+json"},
                meta={
                    "id_from": f"{LEXICON}-{synset['id']}",
                    "rel_prefix": "pwn31",
                    "dont_cache": True,
                },
                callback=self.parse_wd_pwn,
            )

            for rel, keys in synset.get("old_keys", {}).items():
                for k in keys:
                    yield PwnFriendsItem(
                        id_from=f"{LEXICON}-{synset['id']}",
                        id_to=f"{rel}-{k}",
                        rel=f"pwn31_to_{rel}",
                    )

    def parse_wd_pwn(self, response):
        json_response = json.loads(response.body)

        for field in jmespath.search("results.bindings", json_response) or []:
            for k, v in field.items():
                if k == "item":
                    m = re.search(r"(Q\d+)", v["value"])
                    if m:
                        yield PwnFriendsItem(
                            id_from=response.meta["id_from"],
                            id_to=f"wd-{m.group(1)}",
                            rel=f"{response.meta['rel_prefix']}_to_wd",
                        )
                elif k.endswith("_wiki"):
                    yield PwnFriendsItem(
                        id_from=response.meta["id_from"],
                        id_to=v["value"],
                        rel=f"{response.meta['rel_prefix']}_to_{k}",
                    )
