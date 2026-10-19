const form = document.querySelector("#query-form");
const query = document.querySelector("#query");
const token = document.querySelector("#token");
const submit = document.querySelector("#submit");
const result = document.querySelector("#result");
const answer = document.querySelector("#answer");
const citations = document.querySelector("#citations");
const finishReason = document.querySelector("#finish-reason");
const runState = document.querySelector("#run-state");

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    query.value = button.dataset.query;
    updateCount();
    query.focus();
  });
});
query.addEventListener("input", updateCount);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setRunning(true);
  result.hidden = true;
  setSteps(0);
  try {
    setSteps(1);
    const response = await fetch("/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token.value.trim()}` },
      body: JSON.stringify({ query: query.value.trim(), workspace_id: "policies", max_retries: 2 }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload?.error?.message || `HTTP ${response.status}`);
    setSteps(3);
    render(payload);
    setRunning(false);
  } catch (error) {
    showError(error instanceof Error ? error.message : "请求失败");
    submit.disabled = false;
  }
});

function updateCount() { document.querySelector("#char-count").textContent = `${query.value.length} / 8000`; }
function setRunning(running) { submit.disabled = running; runState.textContent = running ? "RUNNING" : "DONE"; runState.className = `badge ${running ? "" : "idle"}`; }
function setSteps(count) { document.querySelectorAll("#steps li").forEach((item, index) => item.classList.toggle("done", index < count)); }

function render(payload) {
  result.hidden = false;
  answer.className = "answer";
  answer.textContent = payload.answer;
  finishReason.textContent = payload.finish_reason;
  finishReason.className = `badge ${payload.finish_reason === "policy_blocked" ? "blocked" : ""}`;
  document.querySelector("#trace-id").textContent = payload.trace_id;
  document.querySelector("#retries").textContent = String(payload.retries);
  document.querySelector("#latency").textContent = `${payload.latency_ms} ms`;
  citations.replaceChildren(...payload.citations.map(citationCard));
  if (!payload.citations.length) citations.textContent = "本次响应没有企业知识库引用。";
}

function citationCard(item) {
  const card = document.createElement("article");
  card.className = "citation";
  const title = document.createElement("strong");
  title.textContent = `${item.title} · ${item.chunk_id}`;
  const snippet = document.createElement("p");
  snippet.textContent = item.snippet;
  card.append(title, snippet);
  return card;
}

function showError(message) {
  result.hidden = false;
  answer.textContent = message;
  answer.className = "answer error-message";
  citations.replaceChildren();
  finishReason.textContent = "error";
  finishReason.className = "badge error";
  runState.textContent = "ERROR";
  runState.className = "badge error";
}

updateCount();
