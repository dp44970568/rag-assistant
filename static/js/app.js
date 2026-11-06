/**
 * RAG AI Assistant Frontend Application
 * High-performance vanilla JS with Server-Sent Events (SSE) streaming.
 */

// Application State
const state = {
  sessionId: "sess_" + Math.random().toString(36).substring(2, 10),
  activeProvider: "offline",
  modelName: "gemini-1.5-flash",
  topK: 4,
  threshold: 0.15,
  docFilter: "",
  isStreaming: false,
  activeAbortController: null,
  documents: [],
  theme: localStorage.getItem("rag_theme") || "dark"
};

// DOM Elements Cache
const elements = {
  chatMessages: document.getElementById("chatMessages"),
  promptInput: document.getElementById("promptInput"),
  sendBtn: document.getElementById("sendBtn"),
  fileInput: document.getElementById("fileInput"),
  dropzone: document.getElementById("dropzone"),
  docList: document.getElementById("docList"),
  docFilterSelect: document.getElementById("docFilterSelect"),
  topKSlider: document.getElementById("topKSlider"),
  topKVal: document.getElementById("topKVal"),
  thresholdSlider: document.getElementById("thresholdSlider"),
  thresholdVal: document.getElementById("thresholdVal"),
  totalDocsBadge: document.getElementById("totalDocsBadge"),
  totalChunksBadge: document.getElementById("totalChunksBadge"),
  providerBadge: document.getElementById("providerBadge"),
  themeToggleBtn: document.getElementById("themeToggleBtn"),
  clearChatBtn: document.getElementById("clearChatBtn"),
  exportChatBtn: document.getElementById("exportChatBtn"),
  loadSampleBtn: document.getElementById("loadSampleBtn"),
  
  // Settings Modal
  settingsModal: document.getElementById("settingsModal"),
  openSettingsBtn: document.getElementById("openSettingsBtn"),
  closeSettingsBtn: document.getElementById("closeSettingsBtn"),
  saveSettingsBtn: document.getElementById("saveSettingsBtn"),
  settingProvider: document.getElementById("settingProvider"),
  settingModel: document.getElementById("settingModel"),
  settingGeminiKey: document.getElementById("settingGeminiKey"),
  settingOpenaiKey: document.getElementById("settingOpenaiKey"),
  settingGroqKey: document.getElementById("settingGroqKey"),
  settingChunkSize: document.getElementById("settingChunkSize"),
  settingChunkOverlap: document.getElementById("settingChunkOverlap"),
  settingSystemPrompt: document.getElementById("settingSystemPrompt"),

  // Chunk Inspector Modal
  chunkModal: document.getElementById("chunkModal"),
  chunkModalTitle: document.getElementById("chunkModalTitle"),
  chunkModalBody: document.getElementById("chunkModalBody"),
  closeChunkModalBtn: document.getElementById("closeChunkModalBtn"),

  // Source Citation Modal
  sourceModal: document.getElementById("sourceModal"),
  sourceModalTitle: document.getElementById("sourceModalTitle"),
  sourceModalBody: document.getElementById("sourceModalBody"),
  closeSourceModalBtn: document.getElementById("closeSourceModalBtn"),
};

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initEventListeners();
  loadInitialStatus();
  autoResizeTextarea();
});

function initTheme() {
  document.documentElement.setAttribute("data-theme", state.theme);
  updateThemeIcon();
}

function toggleTheme() {
  state.theme = state.theme === "dark" ? "light" : "dark";
  localStorage.setItem("rag_theme", state.theme);
  document.documentElement.setAttribute("data-theme", state.theme);
  updateThemeIcon();
}

function updateThemeIcon() {
  if (!elements.themeToggleBtn) return;
  elements.themeToggleBtn.innerHTML = state.theme === "dark" 
    ? `<i data-lucide="sun" class="w-4 h-4"></i>` 
    : `<i data-lucide="moon" class="w-4 h-4"></i>`;
  if (window.lucide) lucide.createIcons();
}

// Event Listeners
function initEventListeners() {
  // Theme
  elements.themeToggleBtn.addEventListener("click", toggleTheme);

  // Send message
  elements.sendBtn.addEventListener("click", handleSendMessage);
  elements.promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  // Sliders
  elements.topKSlider.addEventListener("input", (e) => {
    state.topK = parseInt(e.target.value);
    elements.topKVal.innerText = state.topK;
  });

  elements.thresholdSlider.addEventListener("input", (e) => {
    state.threshold = parseFloat(e.target.value);
    elements.thresholdVal.innerText = state.threshold.toFixed(2);
  });

  // Filter
  elements.docFilterSelect.addEventListener("change", (e) => {
    state.docFilter = e.target.value;
  });

  // File Upload
  elements.dropzone.addEventListener("click", () => elements.fileInput.click());
  elements.fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) uploadFiles(Array.from(e.target.files));
  });

  // Drag and Drop
  elements.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    elements.dropzone.classList.add("dragover");
  });
  elements.dropzone.addEventListener("dragleave", () => {
    elements.dropzone.classList.remove("dragover");
  });
  elements.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    elements.dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadFiles(Array.from(e.dataTransfer.files));
  });

  // Action Buttons
  elements.loadSampleBtn.addEventListener("click", loadSampleData);
  elements.clearChatBtn.addEventListener("click", clearChat);
  elements.exportChatBtn.addEventListener("click", exportChat);

  // Settings Modal
  elements.openSettingsBtn.addEventListener("click", openSettingsModal);
  elements.closeSettingsBtn.addEventListener("click", closeSettingsModal);
  elements.saveSettingsBtn.addEventListener("click", saveSettings);

  // Modals backdrop clicks
  elements.settingsModal.addEventListener("click", (e) => {
    if (e.target === elements.settingsModal) closeSettingsModal();
  });
  elements.chunkModal.addEventListener("click", (e) => {
    if (e.target === elements.chunkModal) elements.chunkModal.classList.remove("active");
  });
  elements.sourceModal.addEventListener("click", (e) => {
    if (e.target === elements.sourceModal) elements.sourceModal.classList.remove("active");
  });

  elements.closeChunkModalBtn.addEventListener("click", () => elements.chunkModal.classList.remove("active"));
  elements.closeSourceModalBtn.addEventListener("click", () => elements.sourceModal.classList.remove("active"));
}

function autoResizeTextarea() {
  elements.promptInput.addEventListener("input", () => {
    elements.promptInput.style.height = "auto";
    elements.promptInput.style.height = Math.min(180, elements.promptInput.scrollHeight) + "px";
  });
}

// Fetch Initial App Status
async function loadInitialStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    
    state.activeProvider = data.provider;
    state.modelName = data.model_name;
    state.topK = data.top_k;
    state.threshold = data.similarity_threshold;

    elements.topKSlider.value = state.topK;
    elements.topKVal.innerText = state.topK;
    elements.thresholdSlider.value = state.threshold;
    elements.thresholdVal.innerText = state.threshold.toFixed(2);

    updateProviderBadge();
    renderDocuments(data.stats.documents || []);
    updateStatsBadges(data.stats);
  } catch (err) {
    console.error("Failed to load initial status:", err);
  }
}

function updateProviderBadge() {
  const p = state.activeProvider.toLowerCase();
  let label = "Offline Synthesizer";
  let dotClass = "amber";

  if (p === "gemini") {
    label = `Gemini (${state.modelName})`;
    dotClass = "status-dot";
  } else if (p === "openai") {
    label = `OpenAI (${state.modelName})`;
    dotClass = "status-dot";
  } else if (p === "groq") {
    label = `Groq (${state.modelName})`;
    dotClass = "status-dot";
  } else if (p === "ollama") {
    label = `Ollama (${state.modelName})`;
    dotClass = "status-dot";
  }

  elements.providerBadge.innerHTML = `<span class="status-dot ${dotClass}"></span> <span>${label}</span>`;
}

function updateStatsBadges(stats) {
  elements.totalDocsBadge.innerText = stats.total_documents || 0;
  elements.totalChunksBadge.innerText = stats.total_chunks || 0;
}

// Documents Management
function renderDocuments(docs) {
  state.documents = docs;
  elements.docList.innerHTML = "";

  // Update filter select
  elements.docFilterSelect.innerHTML = `<option value="">All Knowledge Base (${docs.length})</option>`;

  if (docs.length === 0) {
    elements.docList.innerHTML = `
      <div style="text-align: center; color: var(--text-dim); padding: 16px; font-size: 0.8rem;">
        No documents indexed yet.<br>Upload files or click "Load Sample Knowledge".
      </div>`;
    return;
  }

  docs.forEach((doc) => {
    // Add to filter dropdown
    const opt = document.createElement("option");
    opt.value = doc.name;
    opt.innerText = doc.name;
    if (state.docFilter === doc.name) opt.selected = true;
    elements.docFilterSelect.appendChild(opt);

    // Add to sidebar list
    const item = document.createElement("div");
    item.className = "doc-item";
    item.innerHTML = `
      <div class="doc-info" title="${doc.name}">
        <i data-lucide="file-text" class="w-4 h-4 text-indigo-400 flex-shrink-0"></i>
        <div style="overflow: hidden;">
          <div class="doc-name">${escapeHtml(doc.name)}</div>
          <div class="doc-meta">${doc.chunk_count} chunks • ${formatBytes(doc.char_count)}</div>
        </div>
      </div>
      <div class="doc-actions">
        <button class="btn-icon" title="Inspect Chunks" onclick="inspectDocChunks('${escapeJs(doc.name)}')">
          <i data-lucide="layers" class="w-4 h-4"></i>
        </button>
        <button class="btn-icon danger" title="Delete Document" onclick="deleteDocument('${escapeJs(doc.name)}')">
          <i data-lucide="trash-2" class="w-4 h-4"></i>
        </button>
      </div>
    `;
    elements.docList.appendChild(item);
  });

  if (window.lucide) lucide.createIcons();
}

async function uploadFiles(files) {
  if (!files.length) return;
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));

  elements.dropzone.innerHTML = `<div style="font-size: 0.85rem; color: var(--accent-primary);">Uploading & Chunking...</div>`;

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    renderDocuments(data.current_stats.documents || []);
    updateStatsBadges(data.current_stats);
  } catch (err) {
    alert("Upload failed: " + err.message);
  } finally {
    elements.dropzone.innerHTML = `
      <i data-lucide="upload-cloud" class="dropzone-icon w-7 h-7"></i>
      <div style="font-size: 0.85rem; font-weight: 600;">Drop documents here</div>
      <div style="font-size: 0.72rem; color: var(--text-dim);">PDF, DOCX, TXT, MD, CSV, JSON</div>
    `;
    elements.fileInput.value = "";
    if (window.lucide) lucide.createIcons();
  }
}

async function deleteDocument(docName) {
  if (!confirm(`Are you sure you want to delete '${docName}' from the knowledge base?`)) return;
  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(docName)}`, { method: "DELETE" });
    const data = await res.json();
    renderDocuments(data.stats.documents || []);
    updateStatsBadges(data.stats);
  } catch (err) {
    alert("Delete error: " + err.message);
  }
}

async function loadSampleData() {
  elements.loadSampleBtn.disabled = true;
  elements.loadSampleBtn.innerText = "Loading...";
  try {
    const res = await fetch("/api/load-sample", { method: "POST" });
    const data = await res.json();
    renderDocuments(data.stats.documents || []);
    updateStatsBadges(data.stats);
  } catch (err) {
    alert("Failed to load sample: " + err.message);
  } finally {
    elements.loadSampleBtn.disabled = false;
    elements.loadSampleBtn.innerHTML = `<i data-lucide="sparkles" class="w-3.5 h-3.5"></i> Load Sample Knowledge`;
    if (window.lucide) lucide.createIcons();
  }
}

// Chunk Inspector
window.inspectDocChunks = async function(docName) {
  try {
    const res = await fetch(`/api/chunks/${encodeURIComponent(docName)}`);
    const data = await res.json();
    elements.chunkModalTitle.innerText = `Chunks for: ${data.document} (${data.chunk_count} chunks)`;
    
    elements.chunkModalBody.innerHTML = data.chunks.map((chk, i) => `
      <div style="background: var(--bg-primary); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--text-dim); margin-bottom: 6px;">
          <strong>Chunk #${i + 1} (${chk.chunk_id})</strong>
          <span>${chk.word_count} words • ~${chk.token_estimate} tokens</span>
        </div>
        <div style="font-size: 0.85rem; line-height: 1.5; white-space: pre-wrap; font-family: monospace; background: var(--bg-secondary); padding: 8px; border-radius: 6px;">${escapeHtml(chk.text)}</div>
      </div>
    `).join("");

    elements.chunkModal.classList.add("active");
  } catch (err) {
    alert("Error fetching chunks: " + err.message);
  }
};

// Source Citation Viewer
window.showCitationModal = function(sourceData) {
  elements.sourceModalTitle.innerText = `📄 ${sourceData.doc_name} (Chunk #${sourceData.chunk_index + 1})`;
  elements.sourceModalBody.innerHTML = `
    <div style="display: flex; gap: 10px; margin-bottom: 12px; font-size: 0.8rem;">
      <span class="citation-score">Relevance Match: ${(sourceData.score * 100).toFixed(1)}%</span>
      <span style="color: var(--text-dim);">${sourceData.char_count || sourceData.full_text.length} characters</span>
    </div>
    <div style="background: var(--bg-primary); border: 1px solid var(--border-color); border-radius: 8px; padding: 14px; font-size: 0.9rem; line-height: 1.6; white-space: pre-wrap;">${escapeHtml(sourceData.full_text)}</div>
  `;
  elements.sourceModal.classList.add("active");
};

// Chat Flow & SSE Streaming
async function handleSendMessage() {
  const query = elements.promptInput.value.trim();
  if (!query || state.isStreaming) return;

  // Clear welcome hero if present
  const hero = document.getElementById("welcomeHero");
  if (hero) hero.remove();

  // 1. Append User Message
  appendUserMessage(query);
  elements.promptInput.value = "";
  elements.promptInput.style.height = "26px";

  // 2. Prepare Assistant Bubble
  const { bubble, citationsRow, messageRow } = createAssistantRow();
  const cursorSpan = document.createElement("span");
  cursorSpan.className = "cursor-blink";
  bubble.appendChild(cursorSpan);

  state.isStreaming = true;
  elements.sendBtn.disabled = true;

  let assistantContent = "";
  let retrievedSources = [];

  // 3. Connect to SSE Stream
  try {
    const params = new URLSearchParams({
      query: query,
      session_id: state.sessionId,
      top_k: state.topK,
      threshold: state.threshold,
      doc_filter: state.docFilter || ""
    });

    const response = await fetch(`/api/stream?${params.toString()}`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // keep last incomplete line in buffer

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const jsonStr = line.substring(6).trim();
        if (!jsonStr) continue;

        try {
          const packet = JSON.parse(jsonStr);

          if (packet.type === "sources") {
            retrievedSources = packet.sources || [];
            renderCitations(citationsRow, retrievedSources);
          } else if (packet.type === "token") {
            assistantContent += packet.delta;
            bubble.innerHTML = renderMarkdown(assistantContent);
            bubble.appendChild(cursorSpan);
            scrollToBottom();
          } else if (packet.type === "done") {
            cursorSpan.remove();
          } else if (packet.type === "error") {
            bubble.innerHTML += `<div style="color: var(--accent-rose); margin-top: 8px;">Error: ${escapeHtml(packet.message)}</div>`;
          }
        } catch (parseErr) {
          console.error("SSE JSON parse error:", parseErr, jsonStr);
        }
      }
    }
  } catch (err) {
    bubble.innerHTML = `<div style="color: var(--accent-rose);">Streaming failed: ${escapeHtml(err.message)}</div>`;
  } finally {
    cursorSpan.remove();
    state.isStreaming = false;
    elements.sendBtn.disabled = false;
    elements.promptInput.focus();
    scrollToBottom();
  }
}

function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row user";
  row.innerHTML = `
    <div class="avatar user-av"><i data-lucide="user" class="w-4 h-4"></i></div>
    <div class="message-body">
      <div class="message-bubble">${escapeHtml(text)}</div>
    </div>
  `;
  elements.chatMessages.appendChild(row);
  if (window.lucide) lucide.createIcons();
  scrollToBottom();
}

function createAssistantRow() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  
  const body = document.createElement("div");
  body.className = "message-body";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  const citationsRow = document.createElement("div");
  citationsRow.className = "citations-container";
  citationsRow.style.display = "none";

  body.appendChild(bubble);
  body.appendChild(citationsRow);

  row.innerHTML = `<div class="avatar bot-av"><i data-lucide="bot" class="w-4 h-4"></i></div>`;
  row.appendChild(body);

  elements.chatMessages.appendChild(row);
  if (window.lucide) lucide.createIcons();
  scrollToBottom();

  return { bubble, citationsRow, messageRow: row };
}

function renderCitations(container, sources) {
  if (!sources || !sources.length) return;
  container.style.display = "flex";
  container.innerHTML = `<div style="width: 100%; font-size: 0.72rem; font-weight: 700; color: var(--text-dim); text-transform: uppercase; margin-bottom: 4px;">Cited Sources:</div>`;

  sources.forEach((src) => {
    const pill = document.createElement("button");
    pill.className = "citation-pill";
    pill.innerHTML = `
      <i data-lucide="file-check" class="w-3.5 h-3.5"></i>
      <span>${escapeHtml(src.doc_name)} (#${src.chunk_index + 1})</span>
      <span class="citation-score">${(src.score * 100).toFixed(0)}%</span>
    `;
    pill.addEventListener("click", () => window.showCitationModal(src));
    container.appendChild(pill);
  });

  if (window.lucide) lucide.createIcons();
}

// Markdown Formatter
function renderMarkdown(md) {
  if (window.marked) {
    try {
      return marked.parse(md);
    } catch (e) {}
  }
  // Lightweight native fallback
  return md
    .replace(/^### (.*$)/gim, "<h3>$1</h3>")
    .replace(/^## (.*$)/gim, "<h2>$1</h2>")
    .replace(/^# (.*$)/gim, "<h1>$1</h1>")
    .replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/gim, "<em>$1</em>")
    .replace(/`([^`]+)`/gim, "<code>$1</code>")
    .replace(/^\> (.*$)/gim, "<blockquote>$1</blockquote>")
    .replace(/\n/gim, "<br>");
}

function scrollToBottom() {
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// Settings Modal Management
function openSettingsModal() {
  elements.settingProvider.value = state.activeProvider;
  elements.settingModel.value = state.modelName;
  elements.settingsModal.classList.add("active");
}

function closeSettingsModal() {
  elements.settingsModal.classList.remove("active");
}

async function saveSettings() {
  const provider = elements.settingProvider.value;
  const modelName = elements.settingModel.value;
  const geminiKey = elements.settingGeminiKey.value.trim();
  const openaiKey = elements.settingOpenaiKey.value.trim();
  const groqKey = elements.settingGroqKey.value.trim();
  const chunkSize = parseInt(elements.settingChunkSize.value);
  const chunkOverlap = parseInt(elements.settingChunkOverlap.value);
  const systemPrompt = elements.settingSystemPrompt.value.trim();

  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        llm_provider: provider,
        model_name: modelName,
        gemini_api_key: geminiKey || undefined,
        openai_api_key: openaiKey || undefined,
        groq_api_key: groqKey || undefined,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        system_prompt: systemPrompt || undefined
      })
    });

    const data = await res.json();
    state.activeProvider = data.current_config.llm_provider;
    state.modelName = data.current_config.model_name;

    updateProviderBadge();
    closeSettingsModal();
    alert("Settings saved successfully!");
  } catch (err) {
    alert("Error saving settings: " + err.message);
  }
}

// Clear and Export Chat
async function clearChat() {
  if (!confirm("Clear conversation history?")) return;
  try {
    await fetch(`/api/session/${state.sessionId}`, { method: "DELETE" });
    elements.chatMessages.innerHTML = `
      <div id="welcomeHero" class="welcome-hero">
        <div class="hero-icon"><i data-lucide="bot" class="w-8 h-8"></i></div>
        <h2 class="hero-title">RAG AI Assistant</h2>
        <p class="hero-subtitle">Chat cleared. Upload your documents or click a sample question below to begin!</p>
      </div>
    `;
    if (window.lucide) lucide.createIcons();
  } catch (err) {
    console.error("Clear chat error:", err);
  }
}

async function exportChat() {
  try {
    const res = await fetch(`/api/session/${state.sessionId}/export`);
    const data = await res.json();
    const blob = new Blob([data.markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `rag_conversation_${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert("Export failed: " + err.message);
  }
}

// Helpers
window.askPreset = function(question) {
  elements.promptInput.value = question;
  handleSendMessage();
};

function escapeHtml(str) {
  return (str || "").replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag));
}

function escapeJs(str) {
  return (str || "").replace(/'/g, "\\'").replace(/"/g, '\\"');
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}
