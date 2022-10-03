#!/usr/bin/env node
const wtf = require('wtf_wikipedia')
let args = process.argv.slice(2);

let title = args.join(' ');
let lang = "uk";

if (!title) {
    throw new Error('Usage: wtf буряк');
}


wtf.fetch(title, lang, function(err, doc) {
    if (err) {
        console.error(err);
    }

    console.log(JSON.stringify({
        "json": doc.json(),
        "plaintext": doc.plaintext(),
        "wikidata_id": doc.wikidata(),
    }, null, 0));
})