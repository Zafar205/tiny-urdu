// Check lib/urdu.ts against the Python front end:  npm run parity -- expected.json
// (expected.json comes from a script that runs urdu/frontend.py on the same sentences.)
import { readFileSync } from "node:fs";
import { Frontend } from "../lib/urdu.ts";

const { symbols, cases } = JSON.parse(readFileSync(process.argv[2], "utf8"));
const fe = new Frontend(symbols);
let bad = 0;
for (const c of cases) {
  const got = fe.phonemize(c.text).symbols.join(" "), want = c.symbols.join(" ");
  if (got !== want) { bad++; console.log(`MISMATCH ${c.text}\n  py: ${want}\n  ts: ${got}`); }
}
console.log(`${cases.length - bad}/${cases.length} match`);
process.exit(bad ? 1 : 0);
