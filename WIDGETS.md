# Blog widgets

- LikeBtn and MapMyVisitors appear together below the competition rankings and above the Overview / Results / resource buttons. They sit side by side on desktop and stack on narrow screens.
- LikeBtn uses the provider's normal voting and cancellation rules, with the dislike button hidden. MapMyVisitors loads once when the page opens.
- Both language versions share the LikeBtn identifier `vla-sft-in-practice`. Switching languages recreates only the LikeBtn display and does not reload the visitor map.
- Opening the HTML from disk or localhost does not contact either provider or add test visits.

## Configure the published site

1. Visit [MapMyVisitors](https://mapmyvisitors.com/add), enter the final GitHub Pages URL including the repository path, and choose the **Map** widget. Use automatic width (`w=a`); the page limits the map container to 360px. Copy the `src` URL from the generated `<script id="mapmyvisitors">` tag. Keep the generated `d` value unchanged.
2. LikeBtn can use its standard domain-based embed without a site ID. For a project hosted under a path such as `username.github.io/project/`, register that site in [LikeBtn](https://likebtn.com/en/faq) and supply its `site_id` if you want its statistics isolated from other sites on the same domain. Voting restrictions remain managed by LikeBtn.
3. Run from `WebPage` (replace the sample values with your own; do not deploy the examples):

   ```bash
   python -B source/sync_widgets.py \
     --mapmyvisitors-url 'https://mapmyvisitors.com/map.js?d=YOUR_GENERATED_ID&w=a' \
     --likebtn-site-id 'YOUR_LIKEBTN_SITE_ID'
   ```

   Omit `--likebtn-site-id` to keep domain-based LikeBtn configuration. The script updates `source/widgets.json` and the embedded configuration in `index.html`. Never use another site's widget ID.

4. Publish `index.html` and the existing `assets/` directory to GitHub Pages. Confirm the widget requests succeed on the final HTTPS address and that the provider's statistics correspond to that address.

MapMyVisitors is configured for [the published report](https://yzmyalier.github.io/Lingbot-Challenge-SST/) using its generated Map widget URL. If the URL is cleared, the page shows “Visitor statistics are not available yet”; no example counts or borrowed visitor map are displayed. Provider loading failures display a short message and do not affect the report's charts or language switch.

## Update the local page

After changing widget configuration or code, run:

```bash
python -B source/sync_widgets.py
```

This refreshes the two widget sections, styles, scripts, and their translations in the existing standalone report. It preserves the generated experimental content and does not require the external benchmark repositories used by `source/build_report.py`. The full report builder reads the same widget files when those repositories are available.

Provider documentation: [LikeBtn HTML embed](https://likebtn.com/en/like-button-html), [settings](https://likebtn.com/en/settings), [initialization](https://likebtn.com/en/docs/like-button-methods), [MapMyVisitors HTML integration](https://mapmyvisitors.com/learn/tools/add-visitor-map-html/).
