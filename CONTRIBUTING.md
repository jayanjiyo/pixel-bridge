# Git workflow

## Branches

- No direct pushes to `main`; everything goes through a pull request (the leader turns on
  branch protection in Settings → Branches if the account plan allows it; otherwise it is a team rule).
- One branch per task: `<member>/<area>-<feature>`
  - `m1/core-color`, `m2/a-palette`, `m3/a-dither`, `m4/b-scale2x`
- Keep PRs small: one or two functions per PR is ideal (easier to review, fewer conflicts).

## Daily loop

```bash
git switch main && git pull                       # start from the latest main
git switch -c m2/a-palette                        # new branch for the task
# ... implement, then:
python -m pytest tests/test_module_a_front.py     # your test file
python scripts/check_rules.py                     # OpenCV for I/O only
git add -A && git commit -m "feat(a/palette): median cut and map_to_palette"
git push -u origin m2/a-palette                   # open a PR on GitHub
```

If `main` moved while you worked: `git pull --rebase origin main`, fix conflicts, push again
(`git push --force-with-lease`).

## Pull requests

- Fill in the PR template (test output line, time, before/after image if visual).
- CI runs the rule check and the skeleton tests; both must be green.
- Reviewer: M1 (or any other member if M1 is busy). Merge with **Squash and merge**.
- Deadline = PR open with your tests passing by 23:59 on the day in `docs/PLAN.md`.

## Commit messages

`feat(a/palette): ...`, `fix(b/xbr): ...`, `test(core): ...`, `docs: ...`

## Rules

- Only edit your own files (see README "Project structure"). Need a change elsewhere? Ask the owner.
- Never call OpenCV (other than in `core/io.py`), SciPy, scikit-image, Pillow, etc. inside `pixelbridge/`.
  Built-in library functions may be used only in `tests/` to check our results.
- Test images: our own photos or public domain images only; keep each file under ~1 MB.
- Do not commit `results/`, zip files, or IDE folders (already in `.gitignore`).
