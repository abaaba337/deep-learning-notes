"""Check local Markdown/HTML links and notebook/Python syntax without running cells."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import ast
import json
import re
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp', '.tif', '.tiff'}


def links(text):
    found = []

    class HTMLLinks(HTMLParser):
        def handle_starttag(self, tag, attrs):
            found.extend(v for k, v in attrs if k in ('src', 'href') and v)

    def walk(tokens):
        for token in tokens:
            found.extend(token.attrGet(k) for k in ('src', 'href') if token.attrGet(k))
            if token.type.startswith('html'):
                HTMLLinks().feed(token.content)
            if token.children:
                walk(token.children)

    walk(MarkdownIt().parse(text))
    return found


def check():
    errors = []
    count = 0
    used_images = set()

    def image_reference(url, parent):
        if '\n' in url or re.match(r'^(?:[a-zA-Z][\w+.-]*:|//|#)', url):
            return
        try:
            rel = unquote(urlsplit(url).path).replace('\\', '/')
            if Path(rel).suffix.lower() in IMAGE_EXTENSIONS:
                used_images.add((parent / rel).resolve())
        except ValueError:
            return  # A code string need not be a URL.

    def python_syntax(source, path, label):
        tree = ast.parse(source, filename=label)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                image_reference(node.value, path.parent)

    for path in ROOT.rglob('*'):
        parts = path.relative_to(ROOT).parts
        if len(parts) > 1 and parts[0] != 'scripts' and not re.match(r'^0[1-6] ', parts[0]):
            continue  # Independent checkouts alongside the notes are outside this learning path.
        if any(part.startswith('.') for part in parts):
            continue
        if path.suffix == '.py':
            python_syntax(path.read_text(encoding='utf-8-sig'), path, str(path))
        if path.suffix not in ('.md', '.ipynb'):
            continue
        if path.suffix == '.ipynb':
            nb = json.loads(path.read_text(encoding='utf-8'))
            assert nb['nbformat'] == 4
            ids = [cell['id'] for cell in nb['cells']]
            assert len(ids) == len(set(ids)), f'Duplicate cell IDs: {path}'
            assert path.name == 'notes.ipynb', path
            texts = []
            for i, cell in enumerate(nb['cells']):
                source = ''.join(cell['source'])
                if cell['cell_type'] == 'code':
                    # Notebook line magics are valid IPython, not Python AST syntax.
                    source = '\n'.join(line for line in source.splitlines() if not line.lstrip().startswith(('%', '!')))
                    python_syntax(source, path, f'{path.name}:cell {i}')
                elif cell['cell_type'] == 'markdown':
                    texts.append(source)
        else:
            texts = [path.read_text(encoding='utf-8')]
        for text in texts:
            for url in links(text):
                image_reference(url, path.parent)
                if re.match(r'^(?:[a-zA-Z][\w+.-]*:|//|#)', url):
                    continue
                rel = unquote(urlsplit(url).path).replace('\\', '/')
                target = ROOT / rel.lstrip('/') if rel.startswith('/') else path.parent / rel
                count += 1
                if not target.exists():
                    errors.append(f'{path.relative_to(ROOT)} -> {url}')
    for chapter in ROOT.glob('0*'):
        if not chapter.is_dir():
            continue
        for image in chapter.rglob('*'):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if image.parent != chapter / 'images':
                errors.append(f'Image outside chapter images folder: {image.relative_to(ROOT)}')
            if image.resolve() not in used_images:
                errors.append(f'Unreferenced image: {image.relative_to(ROOT)}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {count} local links; notebook and Python syntax')


if __name__ == '__main__':
    assert links('[a](a.md)\n<img src="b.png">\n`[ignored](x.md)`') == ['a.md', 'b.png']
    check()
