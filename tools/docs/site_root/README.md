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
they share the host of the manifest. The repository URL has no `wellKnown` on
purpose: the directory reads files on github.com through `blob/...?raw=true`,
which often answers 429 or 503, and a failed `wellKnown` makes the whole crawl
fail.

To publish it, clone only the root of `gh-pages` (the branch holds every
version of the docs, more than 2 GB) next to this repository, so that the local
`gh-pages` branch used by `mike` is not touched. From the repository root:

```bash
git clone --depth 1 --branch gh-pages --filter=blob:none --sparse https://github.com/skforecast/skforecast.git ../skforecast-gh-pages
cp funding.json ../skforecast-gh-pages/
cd ../skforecast-gh-pages
git add funding.json
git commit -m "Publish funding.json at the site root"
git push origin gh-pages
```

Then check that it returns the JSON, not an HTML page, and delete
`../skforecast-gh-pages`:

```bash
curl -sSI https://skforecast.org/funding.json
```

Before the next `mike deploy`, run `git fetch origin`: `mike` moves the local
`gh-pages` branch forward to `origin/gh-pages`, but it does not fetch, so
without it the push of the deployment is rejected.
