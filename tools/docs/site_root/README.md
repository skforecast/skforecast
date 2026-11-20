# Files at the root of skforecast.org

Netlify serves https://skforecast.org from the root of the `gh-pages` branch.
`mike` only rewrites the version folders (`0.27.0/`, `latest/`...) and
`versions.json`, so files at the root of that branch are kept between
deployments, but nothing copies them there: copy them by hand after editing
them, like `tools/docs/robots.txt`.

`funding.json`, at the root of this repository, is the
[funding.json](https://fundingjson.org) manifest of the project, listed in the
FLOSS/fund directory from https://skforecast.org/funding.json. The published
copy must stay identical to it. Its skforecast.org URLs are verified because
they share the host of the manifest; the repository URL is verified by
`.well-known/funding-manifest-urls`, which is read from the `main` branch.

To publish it (from the repository root, with a clean working tree):

```bash
git fetch origin gh-pages
git worktree add -B gh-pages ../skforecast-gh-pages origin/gh-pages
cp funding.json ../skforecast-gh-pages/
cd ../skforecast-gh-pages
git add funding.json
git commit -m "Publish funding.json at the site root"
git push origin gh-pages
```

Then check that it returns the JSON, not an HTML page:

```bash
curl -sSI https://skforecast.org/funding.json
```
