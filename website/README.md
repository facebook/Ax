The Ax website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Building (All-in-one)

For convenience we provide a single shell script to convert the tutorials and build the website in one command:
```bash
./scripts/make_docs.sh
```

To also execute the tutorials, run
```bash
./scripts/make_docs.sh -t
```

To generate a static build of the website in the `website/build` directory, run
```bash
./scripts/make_docs.sh -b
```

## Building (manually)

### Notebooks
Ensure necessary dependencies are installed (ideally to your virtual env):
```bash
pip install -e ".[tutorial]"
```

Tutorials can be executed locally using the following script. This is optional for locally building the website and is slow.
```bash
python3 scripts/run_tutorials.py -w .
```

We convert tutorial notebooks to MDX for embedding as docs. This needs to be done before serving the website and can be done by running this script from the project root:

```bash
python3 scripts/convert_ipynb_to_mdx.py --clean
```

### Docusaurus
You need [Node](https://nodejs.org/en/) >= 18.x and
[Yarn](https://yarnpkg.com/en/) in order to build the Ax website.

Switch to the `website` dir from the project root and start the server:
```bash
cd website
yarn
yarn start
```

Open http://localhost:3000 (if doesn't automatically open).

Anytime you change the contents of the page, the page should auto-update.

Note that you may need to switch to the "Next" version of the website documentation to see your latest changes.

### Sphinx
Sphinx is used to generate an API reference from the source file docstrings. In production we use ReadTheDocs to build and host these docs, but they can also be built locally for testing.
```sh
cd sphinx/
make html
```

The build output is in `sphinx/build/` but Sphinx does not provide a server. There are many ways to view the output, here's an example using python:

```sh
cd sphinx/build/
python3 -m http.server 8000
```


## Publishing

The site is hosted on GitHub pages, via the `gh-pages` branch of the Ax
[GitHub repo](https://github.com/facebook/Ax/tree/gh-pages).
The website is automatically built and published from GitHub Actions - see the
[config file](https://github.com/facebook/Ax/blob/main/.github/workflows/publish_website.yml) for details.
