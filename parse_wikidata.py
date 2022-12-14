from wikidata.client import Client
import wn
import mkwikidata
import urllib.error

# wn.download("omw-en31:1.4")

WORDNET_PROPERTY = "P8814"


def get_hyponyms(synset_id):
    """
    Returns hyponyms for given synset id.
    :param synset_id: string
    :return: list
    """
    if "hyponym" not in wn.synset(synset_id).relations():
        return []
    return [el.id for el in wn.synset(synset_id).relations()["hyponym"]]


def get_instance_hyponyms(synset_id):
    """
    Returns instance hyponyms for given synset id.
    :param synset_id: string
    :return: list
    """
    if "instance_hyponym" not in wn.synset(synset_id).relations():
        return []
    return [el.id for el in wn.synset(synset_id).relations()["instance_hyponym"]]


def get_hypernyms(synset_id):
    """
    Returns hypernyms for given synset id.
    :param synset_id: string
    :return: list
    """
    if "hypernyms" not in wn.synset(synset_id).relations():
        return []
    return [el.id for el in wn.synset(synset_id).relations()["hypernyms"]]


def get_pwn_id(wikidata_id):
    client = Client()
    pwn_prop = client.get(WORDNET_PROPERTY)
    try:
        parent_entity = client.get(wikidata_id, load=True)
    except urllib.error.URLError as e:
        parent_entity = e.read().decode("utf8", 'ignore')

    return parent_entity.get(pwn_prop)


def parse_synset_id(s_id):
    return "-".join(s_id.split('-')[-2:])


def get_wikidata_id(synset_id):
    query = build_query(parse_synset_id(synset_id))
    try:
        result = mkwikidata.run_query(query)['results']['bindings']
        if result:
            return result[0]['itemLabel']['value']
        else:
            print(f"Not found wikidata id for {synset_id}")
            return ""
    except ValueError:
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
