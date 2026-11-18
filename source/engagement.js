(() => {
  'use strict';
  const config = JSON.parse(document.getElementById('report-widgets').textContent);
  const messages = JSON.parse(document.getElementById('report-i18n').textContent).ui;
  const likeSlot = document.getElementById('likebtn-slot');
  const mapSlot = document.getElementById('visitor-map');
  const statuses = {
    like: {node: document.getElementById('likebtn-status'), key: ''},
    map: {node: document.getElementById('visitor-map-status'), key: ''}
  };
  const localPreview = !['http:', 'https:'].includes(location.protocol)
    || ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
  let language = 'en', likeLanguage = '', likeTimer, mapTimer;

  function renderStatuses() {
    Object.values(statuses).forEach(({node, key}) => {
      node.textContent = key ? messages[key][language] : '';
    });
  }
  function status(widget, key) {
    statuses[widget].key = key;
    renderStatuses();
  }

  // LikeBtn owns voting, cancellation, persistence and duplicate-vote rules.
  // Recreate only the button when changing language; the article ID stays fixed.
  window.reportLikeBtnEvent = event => {
    if (event.type === 'likebtn.loaded') {
      clearTimeout(likeTimer);
      status('like', '');
    }
  };
  function renderLikeButton() {
    if (localPreview) {
      status('like', 'widgetLocal');
      return;
    }
    const nextLanguage = language === 'zh' ? 'zh_CN' : 'en';
    if (likeLanguage === nextLanguage) return;
    likeLanguage = nextLanguage;
    const button = document.createElement('span');
    button.className = 'likebtn-wrapper';
    const settings = {
      identifier: config.likebtn.identifier,
      lang: nextLanguage,
      theme: 'transparent',
      dislike_enabled: 'false',
      counter_zero_show: 'true',
      counter_frmt: 'comma',
      event_handler: 'reportLikeBtnEvent'
    };
    if (config.likebtn.siteId) settings.site_id = config.likebtn.siteId;
    Object.entries(settings).forEach(([key, value]) => button.setAttribute(`data-${key}`, value));
    likeSlot.replaceChildren(button);
    status('like', 'likeLoading');
    clearTimeout(likeTimer);
    likeTimer = setTimeout(() => status('like', 'likeUnavailable'), 15000);
    if (window.LikeBtn?.init) {
      window.LikeBtn.init();
    } else if (!document.getElementById('likebtn_wjs')) {
      const script = document.createElement('script');
      script.id = 'likebtn_wjs';
      script.async = true;
      script.src = 'https://w.likebtn.com/js/w/widget.js';
      script.onload = () => window.LikeBtn?.init();
      script.onerror = () => {
        clearTimeout(likeTimer);
        status('like', 'likeUnavailable');
        script.remove();
        likeLanguage = '';
      };
      document.head.append(script);
    }
  }

  function loadVisitorMap() {
    if (!config.mapmyvisitors.scriptUrl) {
      status('map', 'mapNotConfigured');
      return;
    }
    if (localPreview) {
      status('map', 'widgetLocal');
      return;
    }
    let url;
    try {
      url = new URL(config.mapmyvisitors.scriptUrl, 'https://mapmyvisitors.com');
      if (url.origin !== 'https://mapmyvisitors.com' || url.pathname !== '/map.js' || !url.searchParams.get('d')) {
        throw new Error('Use the MapMyVisitors Map widget script URL.');
      }
    } catch {
      status('map', 'mapUnavailable');
      return;
    }
    const script = document.createElement('script');
    script.id = 'mapmyvisitors';
    script.src = url.href;
    script.async = true;
    status('map', 'mapLoading');
    mapTimer = setTimeout(() => status('map', 'mapUnavailable'), 15000);
    script.onload = () => {
      clearTimeout(mapTimer);
      status('map', '');
    };
    script.onerror = () => {
      clearTimeout(mapTimer);
      status('map', 'mapUnavailable');
    };
    // Load once at page opening.
    // Re-inserting on language changes could count the same page view again.
    mapSlot.append(script);
  }

  document.addEventListener('report:languagechange', event => {
    language = event.detail.language === 'zh' ? 'zh' : 'en';
    renderLikeButton();
    renderStatuses();
  });
  loadVisitorMap();
})();
