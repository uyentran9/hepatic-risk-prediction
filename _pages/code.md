---
layout: single
title: "Code"
permalink: /code/
author_profile: true
---

<p>This page embeds the fully executed notebook, exported directly from Jupyter — every cell, output, and chart exactly as run.</p>

<div class="code-actions">
  <a class="data-download-btn" href="{{ site.baseurl }}/files/hepatic_risk_outcome_prediction.ipynb" download>⬇ Download .ipynb</a>
  <a class="data-download-btn code-secondary-btn" href="{{ site.baseurl }}/files/notebook_export.html" target="_blank" rel="noopener">↗ Open full-screen</a>
</div>

<div id="notebook-wrap" class="notebook-wrap">
  <iframe id="notebook-frame" src="{{ site.baseurl }}/files/notebook_export.html" loading="lazy"></iframe>
</div>

<style>
.code-actions { margin: 1em 0 1.5em; display: flex; gap: 0.75em; flex-wrap: wrap; }
.data-download-btn {
  display: inline-block;
  padding: 0.5em 1.1em;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff !important;
  background: var(--global-base-color, #2f6f4f);
  border-radius: 4px;
  text-decoration: none !important;
}
.data-download-btn:hover { opacity: 0.85; }
.code-secondary-btn {
  background: transparent;
  color: var(--global-base-color, #2f6f4f) !important;
  border: 1px solid var(--global-base-color, #2f6f4f);
}
.notebook-wrap {
  border: 1px solid var(--global-border-color, #e8e8e8);
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.1);
  overflow: hidden;
  background: #fff;
}
.notebook-wrap iframe {
  width: 100%;
  height: 85vh;
  min-height: 700px;
  border: 0;
  display: block;
}
.notebook-missing {
  padding: 3em 1.5em;
  text-align: center;
  color: var(--global-text-color-light, #888);
}
.notebook-missing code {
  background: var(--global-code-background-color, #f7f7f8);
  padding: 0.15em 0.4em;
  border-radius: 3px;
}
</style>

<script>
(function () {
  var url = '{{ site.baseurl }}/files/notebook_export.html';
  fetch(url, { method: 'HEAD' }).then(function (res) {
    if (!res.ok) throw new Error('missing');
  }).catch(function () {
    document.getElementById('notebook-wrap').innerHTML =
      '<div class="notebook-missing">The notebook export has not been uploaded yet.<br>' +
      'Expected at <code>' + url + '</code>.</div>';
  });
})();
</script>
