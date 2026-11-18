(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const translations = JSON.parse($('#report-i18n').textContent);
  const textTargets = [];
  const walker = document.createTreeWalker(document, NodeFilter.SHOW_COMMENT);
  while (walker.nextNode()) {
    const marker = walker.currentNode;
    if (!marker.data.startsWith('i18n:')) continue;
    const node = marker.nextSibling, key = marker.data.slice(5);
    if (!node || node.nodeType !== Node.TEXT_NODE || !translations.text[key]) continue;
    const english = translations.text[key].en, start = node.data.indexOf(english);
    const leading = node.data.slice(0,start), trailing = node.data.slice(start+english.length);
    textTargets.push({node, key, leading, trailing});
  }
  const attributeTargets = $$('[data-i18n-attrs]').map(node=>({node,keys:JSON.parse(node.dataset.i18nAttrs)}));
  let language = 'en', activeMetric = 'random';
  function message(key, values={}) {
    return translations.ui[key][language].replace(/\{(\w+)\}/g,(_,name)=>values[name]);
  }
  const labels = {n0:'Absolute joint',n1:'SAM multi',n2:'Judged SAM',n4:'Module LR',n5:'Data integrity fix'};
  const chart = $('#main-chart');
  function renderChart(metric) {
    activeMetric = metric;
    chart.innerHTML = '<div class="chart-rows">' + data.mainline.map(row => `<div class="bar-row ${row.id === 'n5' ? 'final' : ''}"><a href="#${row.id}" class="bar-label"><span class="bar-config">${row.id.toUpperCase()} · ${row.step/1000}k</span>${row.id==='n3'?message('mainN3'):labels[row.id]}</a><div class="bar-track" aria-hidden="true"><span class="bar-fill" style="width:${row[metric]}%"></span></div><strong class="bar-value">${row[metric].toFixed(1)}</strong></div>`).join('') + '</div><div class="bar-axis" aria-hidden="true"><span>0</span><span>20</span><span>40</span><span>60</span><span>80</span><span>100</span></div>';
    chart.setAttribute('aria-label', message('chartLabel',{scene:metric === 'random' ? 'Randomized' : 'Clean'}));
    $$('[data-metric]').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.metric===metric)));
  }
  $$('[data-metric]').forEach(button => button.addEventListener('click',()=>renderChart(button.dataset.metric)));

  function renderTasks() {
    const query = $('#task-search').value.trim().toLowerCase();
    const sort = $('#task-sort').value;
    const tasks = data.tasks.filter(t=>t.name.includes(query));
    const gain=t=>t.random_final-t.random_baseline;
    tasks.sort((a,b)=>sort==='gain-desc'?gain(b)-gain(a):sort==='gain-asc'?gain(a)-gain(b):sort==='final-desc'?b.random_final-a.random_final:a.name.localeCompare(b.name));
    const body=$('#task-table');
    body.replaceChildren();
    tasks.forEach(task=>{
      const row=document.createElement('tr');
      const title=document.createElement('th');title.scope='row';title.textContent=task.name;row.append(title);
      ['clean_baseline','clean_final','random_baseline','random_final'].forEach(key=>{const td=document.createElement('td');td.textContent=task[key].toFixed(0);row.append(td)});
      const delta=gain(task);const td=document.createElement('td');td.className=delta>=0?'gain-positive':'gain-negative';td.textContent=(delta>0?'+':'')+delta.toFixed(0);row.append(td);body.append(row);
    });
    if(!tasks.length){const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=6;td.textContent=message('noTasks');tr.append(td);body.append(tr)}
    $('#task-count').textContent=message('taskCount',{count:tasks.length,total:data.tasks.length});
  }
  $('#task-search').addEventListener('input',renderTasks);
  $('#task-sort').addEventListener('change',renderTasks);

  const repairNames=['click_bell','click_alarmclock'];
  const repair=$('#task-repair-figure');
  function renderRepair() {
    repair.innerHTML='<figure class="repair-chart">'+[['random','Randomized'],['clean','Clean']].map(([metric,label])=>{
    const rows=repairNames.map(name=>{
      const task=data.tasks.find(x=>x.name===name);
      const before=task[`${metric}_before`],after=task[`${metric}_after`];
      return `<div class="repair-row"><span>${name}</span><div class="repair-bars" aria-hidden="true"><i style="width:${before}%"></i><i style="width:${after}%"></i></div><b>${before.toFixed(0)} → <em>${after.toFixed(0)}</em></b></div>`;
    }).join('');
    return `<section class="repair-scene" aria-labelledby="repair-${metric}-title"><h4 id="repair-${metric}-title">${message('clickTitle',{scene:label})}</h4>${rows}</section>`;
    }).join('')+`<figcaption><span class="chart-key"><i></i>${message('beforeRepair')}</span><span class="chart-key final"><i></i>${message('afterRepair')}</span></figcaption></figure>`;
  }

  function preferredLanguage() {
    const requested=new URL(location.href).searchParams.get('lang');
    if (requested==='en'||requested==='zh') return requested;
    try { return localStorage.getItem('vla-report-language')==='zh'?'zh':'en'; }
    catch { return 'en'; }
  }
  function setLanguage(next, remember=false) {
    // Keep the same reading position when translated paragraphs change height.
    const top=$('.topbar').getBoundingClientRect().bottom+24;
    const readingAnchor=$$('main h2, main h3, main h4, main p, main figure, main .table-scroll')
      .find(node=>{const rect=node.getBoundingClientRect();return rect.top<=top && rect.bottom>top;});
    const beforeTop=readingAnchor?.getBoundingClientRect().top;
    language=next==='zh'?'zh':'en';
    document.documentElement.lang=language==='zh'?'zh-CN':'en';
    textTargets.forEach(({node,key,leading,trailing})=>{
      node.data=leading+translations.text[key][language]+trailing;
    });
    attributeTargets.forEach(({node,keys})=>Object.entries(keys).forEach(([attr,key])=>node.setAttribute(attr,translations.text[key][language])));
    $$('[data-language]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.language===language)));
    renderChart(activeMetric);renderTasks();renderRepair();
    document.dispatchEvent(new CustomEvent('report:languagechange', {detail: {language}}));
    if (remember) {
      try { localStorage.setItem('vla-report-language',language); } catch {}
      try { const url=new URL(location.href);url.searchParams.set('lang',language);history.replaceState(null,'',url); } catch {}
      if (readingAnchor?.isConnected) window.scrollBy({top:readingAnchor.getBoundingClientRect().top-beforeTop,behavior:'instant'});
    }
  }
  $$('[data-language]').forEach(button=>button.addEventListener('click',()=>setLanguage(button.dataset.language,true)));
  window.addEventListener('popstate',()=>setLanguage(preferredLanguage()));
  setLanguage(preferredLanguage());

  const tocLinks=$$('.contents a[href^="#"]');
  const tocTargets=tocLinks.map(a=>document.getElementById(a.hash.slice(1))).filter(Boolean);
  let queued=false;
  function onScroll(){
    if(queued)return;queued=true;
    requestAnimationFrame(()=>{
      const max=document.documentElement.scrollHeight-innerHeight;
      $('#reading-progress').style.width=(max>0?Math.min(100,scrollY/max*100):0)+'%';
      let active='lineage';
      for(const s of tocTargets){if(s.id!=='top'&&s.getBoundingClientRect().top<=170)active=s.id;}
      tocLinks.forEach(a=>{const selected=a.hash==='#'+active;a.classList.toggle('active',selected);if(selected)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current')});
      queued=false;
    });
  }
  window.addEventListener('scroll',onScroll,{passive:true});window.addEventListener('resize',onScroll);onScroll();
  $('#print-report').addEventListener('click',()=>window.print());
})();
