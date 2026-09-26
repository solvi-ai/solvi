// The use-case library. Each use case is one file in this folder (see receipt.js for the format):
//   id, title, domain, why, fields: [[name, plain-English description]], code (Python rules + QUESTIONS),
//   docs: [{name, text, today?}], altModel? ("receipts": offer the receipts-tuned extractor).
// To add a use case: copy a file, change it, and add it to the list below.
import receipt from "./receipt.js";
import invoice from "./invoice.js";
import contract from "./contract.js";
import nda from "./nda.js";
import lease from "./lease.js";
import offer from "./offer.js";
import insurance from "./insurance.js";
import bol from "./bol.js";
import support from "./support.js";
import dpa from "./dpa.js";
import blank from "./blank.js";

export const USE_CASES = [receipt, invoice, contract, nda, dpa, lease, offer, insurance, bol, support, blank];
