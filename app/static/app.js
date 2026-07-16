const $ = (id) => document.getElementById(id);

let modelsData = [];
let currentJob = null;
let pollTimer = null;

/* ---------- Model list ---------- */
async function loadModels() {
  const res = await fetch("/api/models");
  const data = await res.json();
  modelsData = data.models;
  const select = $("model-select");
  select.innerHTML = "";
  let anyAvailable = false;
  for (const m of modelsData) {
    const opt = document.createElement("option");
    opt.value = m.key;
    opt.textContent = m.display + (m.available ? "" : " (checkpoint missing)");
    opt.disabled = !m.available;
    if (m.available && !anyAvailable) { opt.selected = true; anyAvailable = true; }
    select.appendChild(opt);
  }
  $("model-status").textContent = anyAvailable
    ? ""
    : `No checkpoints found in ${data.checkpoint_dir}`;
  updateAttributes();
  updateRunButtons();
}

function currentModel() {
  return modelsData.find((m) => m.key === $("model-select").value);
}

function updateAttributes() {
  const m = currentModel();
  const select = $("attr-select");
  select.innerHTML = "";
  const attrs = (m && m.attributes) || [];
  if (!attrs.length) {
    const opt = document.createElement("option");
    opt.textContent = "No attributes found";
    opt.disabled = true;
    opt.selected = true;
    select.appendChild(opt);
  } else {
    for (const a of attrs) {
      const opt = document.createElement("option");
      opt.value = a;
      opt.textContent = a.replaceAll("_", " ");
      select.appendChild(opt);
    }
  }
}

/* ---------- Tabs ---------- */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("panel-" + btn.dataset.tab).classList.add("active");
  });
});

/* ---------- Dropzones ---------- */
const files = {};
function setupDropzone(dzId, inputId, key) {
  const dz = $(dzId);
  const input = $(inputId);
  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => setFile(input.files[0]));
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });
  function setFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    files[key] = file;
    dz.querySelector(".dz-preview").src = URL.createObjectURL(file);
    dz.classList.add("has-image");
    updateRunButtons();
  }
}
setupDropzone("dz-interp-1", "file-interp-1", "interp1");
setupDropzone("dz-interp-2", "file-interp-2", "interp2");
setupDropzone("dz-edit", "file-edit", "edit");

function updateRunButtons() {
  const m = currentModel();
  const modelOk = m && m.available;
  $("run-interp").disabled = !(modelOk && files.interp1 && files.interp2);
  $("run-edit").disabled = !(modelOk && files.edit && m.attributes && m.attributes.length);
}

$("model-select").addEventListener("change", () => { updateAttributes(); updateRunButtons(); });

/* ---------- Sliders ---------- */
$("interp-frames").addEventListener("input", (e) => { $("interp-frames-val").textContent = e.target.value; });
$("lamda").addEventListener("input", (e) => { $("lamda-val").textContent = Number(e.target.value).toFixed(1); });

/* ---------- Run ---------- */
$("run-interp").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("model", $("model-select").value);
  fd.append("sampling", $("interp-sampling").value);
  fd.append("num_diffusion_steps", $("interp-steps").value);
  fd.append("skip", $("interp-skip").value);
  fd.append("cfg_src", $("interp-cfg-src").value);
  fd.append("cfg_tar", $("interp-cfg-tar").value);
  fd.append("num_frames", $("interp-frames").value);
  fd.append("image1", files.interp1);
  fd.append("image2", files.interp2);
  await startJob("/api/interpolate", fd);
});

$("run-edit").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("model", $("model-select").value);
  fd.append("attribute", $("attr-select").value);
  fd.append("lamda", $("lamda").value);
  fd.append("num_diffusion_steps", $("edit-steps").value);
  fd.append("skip", $("edit-skip").value);
  fd.append("cfg_src", $("edit-cfg-src").value);
  fd.append("cfg_tar", $("edit-cfg-tar").value);
  fd.append("image", files.edit);
  await startJob("/api/edit", fd);
});

async function startJob(url, formData) {
  hideError();
  $("results-card").classList.add("hidden");
  $("gallery").innerHTML = "";
  setRunning(true);
  try {
    const res = await fetch(url, { method: "POST", body: formData });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    const { job_id } = await res.json();
    currentJob = job_id;
    $("progress-card").classList.remove("hidden");
    poll();
  } catch (err) {
    showError(err.message);
    setRunning(false);
  }
}

function poll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    try {
      const res = await fetch(`/api/job/${currentJob}`);
      const job = await res.json();
      const pct = Math.round(job.progress * 100);
      $("progress-fill").style.width = pct + "%";
      $("progress-pct").textContent = pct + "%";
      $("progress-message").textContent = job.message;
      if (job.status === "done") {
        $("progress-card").classList.add("hidden");
        showResults(job);
        setRunning(false);
      } else if (job.status === "error") {
        $("progress-card").classList.add("hidden");
        showError(job.error || "Generation failed");
        setRunning(false);
      } else {
        poll();
      }
    } catch {
      poll();
    }
  }, 1500);
}

function setRunning(running) {
  $("run-interp").disabled = running;
  $("run-edit").disabled = running;
  if (!running) updateRunButtons();
}

/* ---------- Results ---------- */
function showResults(job) {
  const gallery = $("gallery");
  gallery.innerHTML = "";
  for (const r of job.results) {
    const item = document.createElement("div");
    item.className = "result-item";
    item.innerHTML = `
      <img src="/outputs/${job.id}/${r.file}" alt="${r.label}" />
      <div class="caption">${r.label.replaceAll("_", " ")}</div>
      <a class="dl" href="/api/download/${job.id}/${r.file}" download>⬇ Download</a>`;
    gallery.appendChild(item);
  }
  $("download-all").onclick = () => {
    job.results.forEach((r, i) => {
      setTimeout(() => {
        const a = document.createElement("a");
        a.href = `/api/download/${job.id}/${r.file}`;
        a.download = r.file;
        document.body.appendChild(a);
        a.click();
        a.remove();
      }, i * 300);
    });
  };
  $("results-card").classList.remove("hidden");
}

/* ---------- Errors ---------- */
function showError(msg) {
  const card = $("error-card");
  card.textContent = msg;
  card.classList.remove("hidden");
}
function hideError() { $("error-card").classList.add("hidden"); }

loadModels();
