# DataForge PS-1 submission checklist

## Verified locally

- [x] Interactive artifact opens with a live preset; no blank-state gate.
- [x] Three deterministic, real-computation labs with sub-second controls.
- [x] Fixed-state and explicit-history equality is numerically tested.
- [x] Softmax is labeled as a contrast, not an equivalent operation.
- [x] Limitations, live/synthetic/precomputed labels, and model boundary are visible.
- [x] Six recent primary papers plus the official Pathway derivation are mapped to claims.
- [x] Source/license record and AI-assistance disclosure are present.
- [x] One-page concept summary is approximately 780 extracted words and renders as one page.
- [x] Four-page technical blog PDF is generated.
- [x] README includes setup, architecture, evidence ledger, limitations, and deployment instructions.
- [x] `npm run check` passes 19 tests, strict typechecking, production build, asset checks, and mandatory-file validation.
- [x] Desktop production preview and live controls were inspected.
- [x] Focus trapping, focus restoration, Escape/arrow navigation, canvas alternatives, reduced-motion handling, and responsive CSS are implemented.
- [x] Scoped security/privacy review found no validated vulnerabilities or secrets.

## Public release verification

- [x] Create a public GitHub repository and push this source package.
- [x] Deploy `dist/` to a public, no-sign-in URL.
- [x] Verify both URLs without sign-in.
- [x] Run a 390 px mobile smoke test on the deployed URL.
- [x] Put the verified URLs in `README.md` and `docs/SUBMISSION_COPY.md`.
- [x] Run `npm ci && npm run check` in GitHub Actions before deployment.
- [x] Add final team-member names to the project and disclosure.

## Human submission actions

- [ ] Upload the concept-summary PDF and blog PDF to the exact fields requested by Unstop.
- [ ] Rehearse `docs/DEMO_SCRIPT.md` and the adversarial questions in `docs/JUDGE_DEFENSE.md`.
- [ ] Submit before the live Unstop deadline; capture the confirmation screen.

## Release contents

- Source and tests
- Rebuilt static `dist/`
- `output/pdf/Synapse_Memory_Lab_Concept_Summary.pdf`
- `output/pdf/Synapse_Memory_Lab_Blog.pdf`
- README, architecture, evidence, provenance, defense, and demo materials

Do not claim an official BDH run or reproduced ARC score. Public deployment and mobile verification are complete; Unstop upload, rehearsal, and final submission remain human actions until separately evidenced.
