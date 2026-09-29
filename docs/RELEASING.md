# Releasing

CI builds everything on each push to `main` and uploads it as artifacts; it
never publishes. Pick a green run of the commit to release:

```bash
gh run list --limit 5
```

## PyPI

Bump the version in `pyproject.toml`, `src/deoverlap/__init__.py`,
`core/Cargo.toml` and `bindings/python/Cargo.toml`, push, wait for CI, then:

```bash
git tag -a vX.Y.Z -m "deoverlap X.Y.Z" && git push origin vX.Y.Z
gh run download <run-id> -p 'wheel-*' -p sdist -D dist
twine upload dist/*/*
```

Wheels embed the README, so they must come from a run of the final commit.

## npm

`deoverlap` is a small JavaScript package; each platform's addon is its own
package under `bindings/node/npm/<platform>/`, listed in `deoverlap`'s
`optionalDependencies`, so npm installs only the matching one.

If the Rust code changed, bump `version` in `bindings/node/package.json` and
`bindings/node/Cargo.toml`, run `npm run version` (updates `npm/*`), and set
`optionalDependencies` to the same version. For a README-only release, bump
just `deoverlap` and leave the platform packages as they are.

```bash
npm login
cd bindings/node
npm ci && npm run build          # generates index.js / index.d.ts
gh run download <run-id> -p 'bindings-*' -D artifacts
npm run artifacts                # copies each .node into npm/<platform>/
scripts/publish.sh               # asks for 2FA in the browser per package
```

`scripts/publish.sh` skips what is already on the registry, so re-run it if
it stops halfway. Pass a one-time code as its argument instead of confirming
in the browser.
