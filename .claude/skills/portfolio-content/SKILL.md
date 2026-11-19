---
name: portfolio-content
description: Draft or revise the written content on eunahjung.app — the About paragraph, project descriptions, skills list, or a new Education/Experience section. Interviews the user before writing and never invents facts. Use when the user wants to write, rewrite, expand, or add copy to their portfolio site.
---

# Portfolio content

All copy lives in `web/src/App.jsx`:

| What | Where |
|---|---|
| Project cards | `projects` array, top of file |
| Skills rows | `skills` array |
| About paragraph | inside `<h2>About</h2>` section |
| Header tagline | `.lede` paragraph |

## Process

**1. Find out what they're working on.** If the request doesn't name a section, ask
which one. Don't guess and start drafting.

**2. Interview before writing.** Use `AskUserQuestion` — up to 4 questions per call,
and prefer concrete options over open prompts. For a tone or phrasing decision, put
the actual candidate sentences in the `preview` field so they're comparing real text,
not labels.

Good questions to work through, depending on the section:
- Who is the reader? (recruiters screening for ML roles reads differently from peers)
- What's the single thing they want remembered?
- For a project: what was the decision or finding, not the tech stack?
- Any numbers they can stand behind? (accuracy, sample size, % change)
- Which claims are they *not* comfortable making?

**3. Never fabricate.** Degrees, institutions, dates, employers, job titles, metrics,
and outcomes come from the user only. If a fact is missing, ask for it. Do not write a
plausible placeholder into a resume — a made-up detail that ships is worse than a
missing section. If they want a scaffold to fill in later, mark it with an obvious
`{/* TODO */}` comment and say so.

**4. Draft in their voice, not generic recruiter-speak.** Avoid "passionate about
leveraging", "results-driven", "cutting-edge". Prefer the concrete: what was built,
what it found, what decision it supports. Short sentences. If unsure of their voice,
offer two or three options with previews and let them pick.

**5. Edit, don't rewrite the file.** Keep the existing JSX structure, `className`
values, and CSS classes. New sections reuse the existing `.card` / `.skills` patterns
rather than inventing styles — check `web/src/App.css` before adding CSS.

**6. Verify and hand off.** Run `npm run build` in `web/` to confirm it compiles, then
tell them to push — `deploy-site.yml` builds and rsyncs to the server on any change
under `web/`. The site is live at https://eunahjung.app.
