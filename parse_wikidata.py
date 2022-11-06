from wikidata.client import Client
import wn
import mkwikidata

# wn.download("omw-en31:1.4")

WORDNET_PROPERTY = "P8814"
PREFIX = "omw-en31"


def get_pwn_id(wikidata_id):
    client = Client()
    pwn_prop = client.get(WORDNET_PROPERTY)
    parent_entity = client.get(wikidata_id, load=True)
    return parent_entity.get(pwn_prop)


def get_hypernyms(pwn_id):
    return {get_wikidata_id(el.id): el.lemmas() for el in wn.synset(f"{PREFIX}-{pwn_id}").hypernyms() if
            wn.synset(f"{PREFIX}-{pwn_id}")}


def parse_synset_id(s_id):
    return "-".join(s_id.split('-')[-2:])


def get_hyponyms(pwn_id):
    return {get_wikidata_id(parse_synset_id(el.id)): el.lemmas() for el in wn.synset(f"{PREFIX}-{pwn_id}").hyponyms()}


def get_wikidata_id(synset_id):
    query = build_query(parse_synset_id(synset_id))
    result = mkwikidata.run_query(query)['results']['bindings']
    if result:
        return result[0]['itemLabel']['value']
    print(f"Not found wikidata id for {synset_id}")
    return ""


def build_query(wordnet_id):
    query = f"""
            SELECT DISTINCT ?item ?itemLabel WHERE {{
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }}
              {{
                SELECT DISTINCT ?item WHERE {{
                  ?item p:P8814 ?statement0.
                  ?statement0 (ps:{WORDNET_PROPERTY}) "{wordnet_id}".
                }}
              }}
            }}
        """
    return query


def get_wiki_url(wikidata_id):
    client = Client()
    response = client.get(wikidata_id, load=True)
    if response and "ukwiki" in response.attributes["sitelinks"]:
        return response.attributes["sitelinks"]["ukwiki"]["url"]
    print(f"No Ukrainian url for {wikidata_id}")
    return ""
