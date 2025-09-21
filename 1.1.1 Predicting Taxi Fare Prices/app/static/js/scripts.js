// static/js/scripts.js
document.addEventListener("DOMContentLoaded", () => {
  const uploadBtn = document.getElementById("upload-btn");
  const fileInput = document.getElementById("file-input");
  const spinner = document.getElementById("spinner");
  const resultsArea = document.getElementById("results-area");
  const downloadCsvBtn = document.getElementById("download-csv");
  const downloadJsonBtn = document.getElementById("download-json");

  function showSpinner() {
    spinner.classList.remove("d-none");
  }
  function hideSpinner() {
    spinner.classList.add("d-none");
  }

  // Handle CSV/JSON file upload and prediction
  uploadBtn?.addEventListener("click", async () => {
    const f = fileInput.files[0];
    if (!f) {
      alert("Please select a CSV or JSON file first.");
      return;
    }
    showSpinner();
    const formData = new FormData();
    formData.append("file", f);

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
      const data = await resp.json();
      renderResults(data.results || data || []);
    } catch (err) {
      alert("Prediction failed: " + err.message);
    } finally {
      hideSpinner();
    }
  });

  function renderResults(results) {
    if (!results || results.length === 0) {
      resultsArea.innerHTML =
        "<p class='text-muted'>No results returned.</p>";
      downloadCsvBtn.classList.add("d-none");
      downloadJsonBtn.classList.add("d-none");
      return;
    }

    const keys = Object.keys(results[0]);
    let html =
      "<div class='table-responsive'><table class='table table-sm table-striped'><thead><tr>";
    keys.forEach((k) => (html += `<th>${k}</th>`));
    html += "</tr></thead><tbody>";

    results.forEach((r) => {
      html += "<tr>";
      keys.forEach((k) => {
        let v = r[k];
        if (typeof v === "number" && v !== null) {
          v = Number(v).toFixed(4);
        }
        html += `<td>${v ?? ""}</td>`;
      });
      html += "</tr>";
    });

    html += "</tbody></table></div>";
    resultsArea.innerHTML = html;

    // Enable downloads
    downloadCsvBtn.classList.remove("d-none");
    downloadJsonBtn.classList.remove("d-none");
    downloadCsvBtn.onclick = () => downloadCSV(results);
    downloadJsonBtn.onclick = () => downloadJSON(results);
  }

  function downloadCSV(results) {
    const keys = Object.keys(results[0]);
    const lines = [keys.join(",")];

    results.forEach((r) => {
      const row = keys
        .map((k) => {
          const v = r[k];
          return v == null ? "" : `"${String(v).replace(/"/g, '""')}"`;
        })
        .join(",");
      lines.push(row);
    });

    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadJSON(results) {
    const blob = new Blob([JSON.stringify(results, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.json";
    a.click();
    URL.revokeObjectURL(url);
  }
});
