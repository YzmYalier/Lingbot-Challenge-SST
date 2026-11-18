#!/usr/bin/env python3
"""Sync widget integration into the existing report without benchmark sources.

Run: python -B source/sync_widgets.py
Optional: --mapmyvisitors-url 'https://mapmyvisitors.com/map.js?d=...&w=a'
          --likebtn-site-id 'SITE_ID'

Only use the MapMyVisitors URL generated for this report's published address.
The original full build_report.py also reads the same source files.
"""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, parse_qs

from i18n import BilingualHTML

HERE = Path(__file__).resolve().parent
SITE = HERE.parent


def json_for_html(value):
    return json.dumps(value, ensure_ascii=False).replace('</', '<\\/')


def replace_once(source, pattern, replacement):
    result, count = re.subn(pattern, lambda match: replacement, source, flags=re.S)
    if count != 1:
        raise ValueError(f'Expected one match, found {count}: {pattern}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mapmyvisitors-url', help='The src URL from your generated Map widget embed; empty clears it.')
    parser.add_argument('--likebtn-site-id', help='Optional LikeBtn site_id, for example for project-path sites.')
    args = parser.parse_args()
    config = json.loads((HERE / 'widgets.json').read_text())
    if args.mapmyvisitors_url is not None:
        url = args.mapmyvisitors_url.strip().replace('&amp;', '&')
        if url.startswith('//'):
            url = 'https:' + url
        parts = urlsplit(url)
        if url and (parts.scheme != 'https' or parts.netloc != 'mapmyvisitors.com'
                    or parts.path != '/map.js' or not parse_qs(parts.query).get('d')):
            parser.error('Use the https://mapmyvisitors.com/map.js?d=... URL from your own Map widget.')
        config['mapmyvisitors']['scriptUrl'] = url
    if args.likebtn_site_id is not None:
        config['likebtn']['siteId'] = args.likebtn_site_id.strip()

    page = (SITE / 'index.html').read_text()
    template = (HERE / 'report.template.html').read_text()
    translations = json.loads((HERE / 'translations.en.json').read_text())
    payload_pattern = r'<script id="report-i18n" type="application/json">(.*?)</script>'
    payload = json.loads(re.search(payload_pattern, page, re.S)[1])

    pattern = r'<div class="hero-engagement">.*?</div>(?=\s*<div class="hero-actions">)'
    fragment = re.search(pattern, template, re.S)[0]
    compiler = BilingualHTML(translations)
    compiler.feed(fragment)
    compiler.close()
    compiled = ''.join(compiler.output)
    payload['text'].update(compiler.messages)
    if re.search(pattern, page, re.S):
        page = replace_once(page, pattern, compiled)
    else:
        # Relocate the earlier article-end widgets, keeping only one of each.
        for class_name in ('reader-feedback', 'visitor-section'):
            page = replace_once(page, rf'<section class="{class_name}".*?</section>\s*', '')
        anchor = '<div class="hero-actions">'
        if page.count(anchor) != 1:
            raise ValueError(f'Missing or duplicate insertion anchor: {anchor}')
        page = page.replace(anchor, compiled + '\n    ' + anchor, 1)

    page = replace_once(page, r'<style>.*?</style>(?=\s*</head>)', '<style>' + (HERE / 'report.css').read_text() + '</style>')
    page = replace_once(page, r'<script(?: id="report-script")?>.*?</script>',
                        '<script id="report-script">' + (HERE / 'report.js').read_text() + '</script>')
    payload['ui'] = json.loads((HERE / 'messages.json').read_text())
    page = replace_once(page, payload_pattern,
                        '<script id="report-i18n" type="application/json">' + json_for_html(payload) + '</script>')

    for script_id, contents, script_type in (
        ('report-widgets', json_for_html(config), ' type="application/json"'),
        ('report-widget-script', (HERE / 'engagement.js').read_text(), ''),
    ):
        replacement = f'<script id="{script_id}"{script_type}>{contents}</script>'
        pattern = rf'<script id="{script_id}"[^>]*>.*?</script>'
        if re.search(pattern, page, re.S):
            page = replace_once(page, pattern, replacement)
        else:
            page = page.replace('<script id="report-script">', replacement + '\n  <script id="report-script">', 1)

    (HERE / 'widgets.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    (SITE / 'index.html').write_text(page)
    print('Synced widget sections, CSS, JavaScript, translations and configuration into index.html.')
    if not config['mapmyvisitors']['scriptUrl']:
        print('MapMyVisitors is not activated yet: supply the Map widget URL for the published site.')


if __name__ == '__main__':
    main()
