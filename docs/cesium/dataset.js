/* Display the identity of the data installed by the static-site build. */
(function () {
  const repo = "PhD-Sergio/data-ntn-topological-evaluation";
  window.setLEOPathDataset = function (data, fallback = "Local examples · unversioned") {
    if (data && (data.repository !== repo || !/^[0-9a-f]{40}$/.test(data.commit) || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(data.tag)))
      throw new Error("Invalid dataset metadata");
    const label = document.getElementById("datasetVersion");
    if (!label) return;
    label.removeAttribute("href");
    label.removeAttribute("title");
    const history = document.getElementById("datasetReleases");
    if (history) history.hidden = true;
    if (!data) { label.textContent = fallback; return; }
    label.textContent = data.preview ? `Dataset preview · ${data.commit.slice(0, 7)}` : `Dataset ${data.tag} · ${data.commit.slice(0, 7)}`;
    label.href = data.preview ? `https://github.com/${repo}` : `https://github.com/${repo}/releases/tag/${encodeURIComponent(data.tag)}`;
    label.title = data.scope || "";
    if (history) history.hidden = false;
  };
  window.leopathDatasetReady = (async function () {
    try {
      const response = await fetch("dataset.json", {cache: "no-store"});
      if (response.status === 404) return null;
      if (!response.ok) throw new Error("Dataset metadata unavailable");
      const data = await response.json();
      window.setLEOPathDataset(data);
      return data;
    } catch (error) {
      window.leopathDatasetError = true;
      window.setLEOPathDataset(null, "Dataset version unavailable");
      return null;
    }
  })();
})();
