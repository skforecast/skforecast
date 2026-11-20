# Files at the root of skforecast.org

Netlify serves https://skforecast.org from the root of the `gh-pages` branch.
`mike` only rewrites the version folders (`0.27.0/`, `latest/`...) and
`versions.json`, so files at the root of that branch are kept between
deployments, but nothing copies them there: copy them by hand after editing
them, like `tools/docs/robots.txt`.

| Source in this repository | Path in `gh-pages` | Served at |
| :-- | :-- | :-- |
| `funding.json` (repository root) | `funding.json` | https://skforecast.org/funding.json |
| `tools/docs/site_root/well-known/funding-manifest-urls.txt` | `well-known/funding-manifest-urls.txt` | https://skforecast.org/.well-known/funding-manifest-urls |
| `tools/docs/site_root/_redirects` | `_redirects` | (Netlify rules) |

`funding.json` is the [funding.json](https://fundingjson.org) manifest of the
project, listed in the FLOSS/fund directory. The repository root keeps the
source; the published copy must stay identical to it.

`funding-manifest-urls` lists the manifest URLs allowed to solicit funding for
the URLs of this host. Netlify does not publish files or folders whose name
starts with a dot (`.nojekyll` and `.gitignore` in `gh-pages` return 404), so
the file lives in `well-known/` and `_redirects` serves it at
`/.well-known/funding-manifest-urls`; its `.txt` extension makes Netlify send
it as `text/plain`. `.well-known/funding-manifest-urls` at the repository root
is a different file: it verifies the GitHub repository URL of the manifest, and
it is read from the `main` branch.

To publish (from the repository root, with a clean working tree):

```bash
git fetch origin gh-pages
git worktree add -B gh-pages ../skforecast-gh-pages origin/gh-pages
cp funding.json ../skforecast-gh-pages/
cp -r tools/docs/site_root/well-known tools/docs/site_root/_redirects ../skforecast-gh-pages/
cd ../skforecast-gh-pages
git add funding.json well-known _redirects
git commit -m "Publish funding.json and its well-known file at the site root"
git push origin gh-pages
```

Then check that both URLs return the files, not an HTML page:

```bash
curl -sSI https://skforecast.org/funding.json
curl -sS https://skforecast.org/.well-known/funding-manifest-urls
```
