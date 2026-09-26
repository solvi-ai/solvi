// node tests/export_usecases.mjs > usecases.json : the use-case library as JSON (texts NFC-normalized, as in the page)
import { USE_CASES } from "../usecases/index.js";
const out = USE_CASES.map((u) => ({ ...u, docs: u.docs.map((d) => ({ ...d, text: d.text.normalize("NFC") })) }));
process.stdout.write(JSON.stringify(out));
