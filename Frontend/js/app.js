const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const sendBtnText = document.getElementById("sendBtnText");
const sendSpinner = document.getElementById("sendSpinner");
const answerSection = document.getElementById("answerSection");
const answerText = document.getElementById("answerText");
const metaInfo = document.getElementById("metaInfo");
const sourcesList = document.getElementById("sourcesList");
const sourcesEmpty = document.getElementById("sourcesEmpty");
const statusPill = document.getElementById("statusPill");

function setLoading(isLoading) {
  sendBtn.disabled = isLoading;
  sendSpinner.classList.toggle("d-none", !isLoading);
  sendBtnText.textContent = isLoading ? "Searching..." : "Send Question";
  statusPill.textContent = isLoading ? "Processing" : "Ready";
}

function formatPage(source) {
  if (source.page_number === source.page_end) {
    return `p. ${source.page_number}`;
  }
  return `pp. ${source.page_number}-${source.page_end}`;
}

function renderSources(sources) {
  sourcesList.innerHTML = "";
  if (!sources.length) {
    sourcesEmpty.classList.remove("d-none");
    return;
  }

  sourcesEmpty.classList.add("d-none");

  sources.forEach((source, index) => {
    const item = document.createElement("article");
    item.className = "source-item";
    item.innerHTML = `
      <div class="d-flex justify-content-between align-items-start gap-2">
        <div>
          <div class="fw-semibold">Source ${index + 1}</div>
          <div class="small text-muted">${source.document_name}</div>
        </div>
        <span class="source-score">${source.similarity_score.toFixed(3)}</span>
      </div>
      <div class="small mt-2"><strong>Section:</strong> ${source.section}</div>
      <div class="small"><strong>Page:</strong> ${formatPage(source)}</div>
      <div class="source-excerpt">${source.excerpt}</div>
    `;
    sourcesList.appendChild(item);
  });
}

async function submitQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }

  setLoading(true);
  answerSection.classList.remove("d-none");
  answerText.textContent = "Retrieving evidence and generating answer...";
  metaInfo.textContent = "";
  renderSources([]);

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Request failed");
    }

    answerText.textContent = data.answer;
    metaInfo.textContent = `Context words used: ${data.context_used_words}`;
    renderSources(data.sources || []);
  } catch (error) {
    answerText.textContent = `Error: ${error.message}`;
  } finally {
    setLoading(false);
  }
}

sendBtn.addEventListener("click", submitQuestion);

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitQuestion();
  }
});

document.querySelectorAll(".sample-btn").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    questionInput.focus();
  });
});
