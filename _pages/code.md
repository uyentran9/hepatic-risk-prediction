---
layout: single
title: "Code"
permalink: /code/
author_profile: true
---

This page provides the original Jupyter Notebook and its executed HTML export for the **Hepatic Risk Outcome Prediction** project. The source notebook contains the Python implementation, while the HTML export allows readers to review saved code cells, explanations, outputs, tables, and figures in a browser.

---

## Jupyter Notebook — Executable Source

The original `.ipynb` file contains the data preparation, machine-learning pipelines, model training, validation, evaluation, and analysis code. Download it to inspect or reproduce the workflow in a Jupyter environment.

<div class="code-actions">
  <a class="code-btn code-primary-btn"
     href="{{ '/files/uyen_hepatic_risk.ipynb' | relative_url }}"
     download>
    ⬇ Download .ipynb
  </a>

  <a class="code-btn code-secondary-btn"
     href="https://github.com/uyentran9/hepatic-risk-prediction/blob/main/files/uyen_hepatic_risk.ipynb"
     target="_blank"
     rel="noopener noreferrer">
    ↗ View on GitHub
  </a>
</div>

---

## Executed Notebook — HTML Preview

The HTML export provides a browser-readable version of the notebook, preserving its saved code cells, Markdown explanations, experimental outputs, tables, and visualizations without requiring the notebook to be rerun. Scroll inside the preview below to explore the full notebook.

<div class="code-actions">
  <a class="code-btn code-primary-btn"
     href="{{ '/files/Kaggle1_Uyen8.html' | relative_url }}"
     download>
    ⬇ Download HTML
  </a>

  <a class="code-btn code-secondary-btn"
     href="{{ '/files/Kaggle1_Uyen8.html' | relative_url }}"
     target="_blank"
     rel="noopener noreferrer">
    ↗ Open full-screen
  </a>
</div>

<div class="notebook-wrap">
  <iframe
    src="{{ '/files/Kaggle1_Uyen8.html' | relative_url }}"
    title="Hepatic Risk Outcome Prediction — Executed Jupyter Notebook HTML Preview"
    loading="lazy"
    scrolling="yes">
  </iframe>
</div>

<style>
/* Layout for download and view buttons */
.code-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75em;
  margin: 1em 0 1.5em;
}

/* Shared button styling */
.code-btn {
  display: inline-block;
  padding: 0.5em 1.1em;
  font-size: 0.9rem;
  font-weight: 600;
  border-radius: 4px;
  text-decoration: none !important;
  transition: background-color 0.2s ease, border-color 0.2s ease;
}

/* Primary action: solid navy */
.code-primary-btn,
.code-primary-btn:visited {
  background: #1f4d7a !important;
  color: #ffffff !important;
  border: 1px solid #1f4d7a !important;
}

/* Secondary action: light blue with navy outline */
.code-secondary-btn,
.code-secondary-btn:visited {
  background: #eef5fb !important;
  color: #1f4d7a !important;
  border: 1px solid #1f4d7a !important;
}

/* Hover and keyboard-focus states */
.code-primary-btn:hover,
.code-primary-btn:focus-visible {
  background: #173b5e !important;
  color: #ffffff !important;
  border-color: #173b5e !important;
}

.code-secondary-btn:hover,
.code-secondary-btn:focus-visible {
  background: #dcebf7 !important;
  color: #173b5e !important;
  border-color: #173b5e !important;
}

/* Scrollable HTML notebook preview */
.notebook-wrap {
  border: 1px solid var(--global-border-color, #e8e8e8);
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  background: #ffffff;
}

.notebook-wrap iframe {
  display: block;
  width: 100%;
  height: 85vh;
  min-height: 700px;
  border: 0;
  background: #ffffff;
}
</style>

<p style="margin-top:24px;">
  <a href="{{ '/' | relative_url }}"
     style="background:#1f4d7a;color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;">
    🏠 Back to Home
  </a>
</p>

Thank you for visiting — and for being part of my journey 🌱
