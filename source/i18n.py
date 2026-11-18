"""Compile English HTML and retain exact Chinese text for an in-place language switch."""
from hashlib import sha1
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
LOCALIZABLE = re.compile(r'[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]')
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}


class BilingualHTML(HTMLParser):
    def __init__(self, translations):
        super().__init__(convert_charrefs=True)
        self.translations = translations
        self.output, self.stack, self.messages = [], [], {}

    def translation(self, source):
        if source not in self.translations:
            raise ValueError(f'Missing English translation: {source!r}')
        key = sha1(source.encode()).hexdigest()[:12]
        self.messages[key] = {'en': self.translations[source], 'zh': source}
        return key, self.translations[source]

    def start(self, tag, attrs, closed=False):
        raw = self.get_starttag_text()
        excluded = any(skip for _, skip in self.stack) or any(k == 'data-no-i18n' for k, _ in attrs)
        localized_attrs = {}
        if not excluded:
            for name, value in attrs:
                if name in ('aria-label', 'title', 'placeholder', 'content') and value and LOCALIZABLE.search(value):
                    key, translated = self.translation(value)
                    pattern = rf'(\b{re.escape(name)}\s*=\s*)([\"\'])(.*?)\2'
                    raw = re.sub(pattern, lambda m: m[1]+'"'+escape(translated, quote=True)+'"', raw, count=1, flags=re.S)
                    localized_attrs[name] = key
        if localized_attrs:
            ending = '/>' if raw.endswith('/>') else '>'
            raw = raw[:-len(ending)] + ' data-i18n-attrs="' + escape(json.dumps(localized_attrs), quote=True) + '"' + ending
        self.output.append(raw)
        if not closed and tag not in VOID:
            self.stack.append((tag, excluded))

    def handle_starttag(self, tag, attrs):
        self.start(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self.start(tag, attrs, closed=True)

    def handle_endtag(self, tag):
        self.output.append(f'</{tag}>')
        for i in range(len(self.stack)-1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, value):
        if any(tag in ('script', 'style') for tag, _ in self.stack):
            self.output.append(value)
            return
        core = value.strip()
        if core and LOCALIZABLE.search(core) and not any(skip for _, skip in self.stack):
            key, translated = self.translation(core)
            # Comments work inside HTML and SVG text without changing layout or semantics.
            leading, trailing = value[:len(value)-len(value.lstrip())], value[len(value.rstrip()):]
            self.output.append(f'<!--i18n:{key}-->'+escape(leading+translated+trailing, quote=False))
        else:
            self.output.append(escape(value, quote=False))

    def handle_comment(self, value):
        self.output.append(f'<!--{value}-->')

    def handle_decl(self, value):
        self.output.append(f'<!{value}>')


def bilingual_html(source):
    translations = json.loads((HERE/'translations.en.json').read_text())
    compiler = BilingualHTML(translations)
    compiler.feed(source)
    compiler.close()
    messages = json.loads((HERE/'messages.json').read_text())
    payload = json.dumps({'text': compiler.messages, 'ui': messages}, ensure_ascii=False).replace('</', '<\\/')
    return ''.join(compiler.output).replace('@@I18N@@', payload)
