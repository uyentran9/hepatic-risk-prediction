---
layout: single
title: "Technical Report"
permalink: /report/
author_profile: true
---

<p>The full technical report documenting the problem formulation, data analysis, modeling approach, experimental results, and limitations of the Hepatic Risk Outcome Prediction project.</p>

<div class="report-actions">
  <a class="report-btn" href="{{ site.baseurl }}/files/HepaticRisk_TechnicalReport_UyenTran.pdf" download>⬇ Download PDF</a>
  <a class="report-btn report-secondary-btn" href="{{ site.baseurl }}/files/HepaticRisk_TechnicalReport_UyenTran.pdf" target="_blank" rel="noopener">↗ Open full-screen</a>
</div>

<div id="report-wrap" class="report-wrap">
  <iframe id="report-frame" src="{{ site.baseurl }}/files/HepaticRisk_TechnicalReport_UyenTran.pdf" loading="lazy"></iframe>
</div>

<style>
.report-actions { margin: 1em 0 1.5em; display: flex; gap: 0.75em; flex-wrap: wrap; }
.report-btn {
  display: inline-block;
  padding: 0.5em 1.1em;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff !important;
  background: var(--global-base-color, #2f6f4f);
  border-radius: 4px;
  text-decoration: none !important;
}
.report-btn:hover { opacity: 0.85; }
.report-secondary-btn {
  background: transparent;
  color: var(--global-base-color, #2f6f4f) !important;
  border: 1px solid var(--global-base-color, #2f6f4f);
}
.report-wrap {
  border: 1px solid var(--global-border-color, #e8e8e8);
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.1);
  overflow: hidden;
  background: #fff;
}
.report-wrap iframe {
  width: 100%;
  height: 90vh;
  min-height: 700px;
  border: 0;
  display: block;
}
.report-missing {
  padding: 3em 1.5em;
  text-align: center;
  color: var(--global-text-color-light, #888);
}
.report-missing code {
  background: var(--global-code-background-color, #f7f7f8);
  padding: 0.15em 0.4em;
  border-radius: 3px;
}
</style>

<script>
(function () {
  var url = '{{ site.baseurl }}/files/HepaticRisk_TechnicalReport_UyenTran.pdf';
  fetch(url, { method: 'HEAD' }).then(function (res) {
    if (!res.ok) throw new Error('missing');
  }).catch(function () {
    document.getElementById('report-wrap').innerHTML =
      '<div class="report-missing">The technical report PDF has not been uploaded yet.<br>' +
      'Expected at <code>' + url + '</code>.</div>';
  });
})();
</script>
