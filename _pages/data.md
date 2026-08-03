---
layout: single
title: "Dataset"
permalink: /data/
author_profile: true
---

<p>Three files are used in this competition: the training set, the test set, and a sample submission file showing the required output format. Each preview below shows the first 20 rows and every column. Use the download buttons to get the full files.</p>

<div class="data-note">
  <strong>Note:</strong> competition data is redistributed here only if permitted by the AIO competition rules. If a file below fails to load, it has not been uploaded to <code>/files/</code> yet.
</div>

<section class="data-block">
  <h2>Training set — <code>aio26_train.csv</code></h2>
  <p>12,000 rows, 20 columns (18 predictors, <code>id</code>, and the target <code>Status</code>).</p>
  <a class="data-download-btn" href="{{ site.baseurl }}/files/aio26_train.csv" download>⬇ Download aio26_train.csv</a>
  <div class="data-table-wrap">
    <table id="train-table"><tbody><tr><td>Loading preview…</td></tr></tbody></table>
  </div>
</section>

<section class="data-block">
  <h2>Test set — <code>aio26_test.csv</code></h2>
  <p>10,000 rows, 19 columns (same predictors as training, no <code>Status</code>).</p>
  <a class="data-download-btn" href="{{ site.baseurl }}/files/aio26_test.csv" download>⬇ Download aio26_test.csv</a>
  <div class="data-table-wrap">
    <table id="test-table"><tbody><tr><td>Loading preview…</td></tr></tbody></table>
  </div>
</section>

<section class="data-block">
  <h2>Sample submission — <code>sample_submission.csv</code></h2>
  <p>Required output format: <code>id</code>, <code>Status_C</code>, <code>Status_CL</code>, <code>Status_D</code>.</p>
  <a class="data-download-btn" href="{{ site.baseurl }}/files/sample_submission.csv" download>⬇ Download sample_submission.csv</a>
  <div class="data-table-wrap">
    <table id="sample-table"><tbody><tr><td>Loading preview…</td></tr></tbody></table>
  </div>
</section>

<style>
.data-note {
  margin: 1em 0 2em;
  padding: 0.75em 1em;
  font-size: 0.9rem;
  color: var(--global-text-color-light, #666);
  background: var(--global-code-background-color, #f7f7f8);
  border-left: 3px solid var(--global-border-color, #e8e8e8);
  border-radius: 4px;
}
.data-block { margin-bottom: 3em; }
.data-download-btn {
  display: inline-block;
  margin: 0.5em 0 1em;
  padding: 0.5em 1.1em;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff !important;
  background: var(--global-base-color, #2f6f4f);
  border-radius: 4px;
  text-decoration: none !important;
}
.data-download-btn:hover { opacity: 0.85; }
.data-table-wrap {
  max-width: 100%;
  overflow-x: auto;
  border: 1px solid var(--global-border-color, #e8e8e8);
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.data-table-wrap table {
  border-collapse: collapse;
  width: max-content;
  min-width: 100%;
  font-size: 0.82rem;
  white-space: nowrap;
}
.data-table-wrap th, .data-table-wrap td {
  padding: 0.4em 0.8em;
  border-bottom: 1px solid var(--global-border-color, #e8e8e8);
  text-align: left;
}
.data-table-wrap thead th {
  position: sticky;
  top: 0;
  background: var(--global-base-color, #2f6f4f);
  color: #fff;
  font-weight: 600;
  z-index: 1;
}
.data-table-wrap tbody tr:nth-child(even) {
  background: var(--global-code-background-color, #f7f7f8);
}
.data-error {
  padding: 1em;
  font-size: 0.85rem;
  color: var(--global-text-color-light, #888);
}
</style>

<script>
(function () {
  var ROWS_TO_SHOW = 20;

  function parseCSV(text) {
    var rows = [];
    var row = [];
    var field = '';
    var inQuotes = false;
    for (var i = 0; i < text.length; i++) {
      var c = text[i];
      if (inQuotes) {
        if (c === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; }
          else { inQuotes = false; }
        } else { field += c; }
      } else {
        if (c === '"') { inQuotes = true; }
        else if (c === ',') { row.push(field); field = ''; }
        else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
        else if (c === '\r') { /* skip */ }
        else { field += c; }
      }
    }
    if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
    return rows.filter(function (r) { return r.length > 1 || (r.length === 1 && r[0] !== ''); });
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function renderTable(tableId, rows) {
    var table = document.getElementById(tableId);
    if (!rows.length) {
      table.innerHTML = '<tbody><tr><td>No data found.</td></tr></tbody>';
      return;
    }
    var header = rows[0];
    var body = rows.slice(1, 1 + ROWS_TO_SHOW);
    var html = '<thead><tr>' + header.map(function (h) {
      return '<th>' + escapeHtml(h) + '</th>';
    }).join('') + '</tr></thead><tbody>';
    body.forEach(function (r) {
      html += '<tr>' + r.map(function (cell) {
        return '<td>' + (cell === '' ? '<em>—</em>' : escapeHtml(cell)) + '</td>';
      }).join('') + '</tr>';
    });
    html += '</tbody>';
    table.innerHTML = html;
  }

  function loadTable(url, tableId) {
    fetch(url)
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.text();
      })
      .then(function (text) {
        renderTable(tableId, parseCSV(text));
      })
      .catch(function () {
        document.getElementById(tableId).outerHTML =
          '<div class="data-error">This file has not been uploaded to <code>' + url + '</code> yet.</div>';
      });
  }

  var base = '{{ site.baseurl }}';
  loadTable(base + '/files/aio26_train.csv', 'train-table');
  loadTable(base + '/files/aio26_test.csv', 'test-table');
  loadTable(base + '/files/sample_submission.csv', 'sample-table');
})();
</script>
