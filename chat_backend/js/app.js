/**
 * AlgoChat v3 — Main Application
 * Gemini-style chat · Workflow nodes · Tab results · File grid · Theme toggle
 */

// ══════════════════════════════════════════
// STATE
// ══════════════════════════════════════════
const App = {
  conversations: [],
  currentConvId: null,
  selectedAlgo: null,
  selectedFiles: [],       // File objects
  selectedFileIndices: [],  // indices into selectedFiles (for multi-select)
  algorithms: [],
  algoCategories: [],
  activeCategory: null,
  chartInstances: {},       // id -> Chart instance
  previewChartInstance: null,
  isProcessing: false,
};

// ══════════════════════════════════════════
// INIT
// ══════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
  // Wave interference intro will handle splash → particles transition
  // The callback _onIntroComplete is set below

  // Load theme
  const savedTheme = localStorage.getItem('algochat_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeLabel(savedTheme);

  // Load algorithms
  App.algorithms = await API.getAlgorithms();
  App.algoCategories = [...new Set(App.algorithms.map(a => a.category))];
  renderAlgoFilterTags();
  renderAlgorithmList();
  renderWelcomeHints();

  // Populate inline algo selector
  populateAlgoSelector();

  // Load conversations from localStorage
  loadConversations();
  renderConversationList();

  // Init particles
  if (localStorage.getItem('algochat_particles') !== 'false') {
    initParticles();
  }

  // Bind all events
  bindEvents();

  // Wave intro complete callback: hide splash overlay, reveal main UI
  window._onIntroComplete = () => {
    const splash = document.getElementById('splashScreen');
    if (splash) {
      splash.classList.add('hidden');
      setTimeout(() => splash.remove(), 600);
    }
  };
});

// ══════════════════════════════════════════
// EVENT BINDING
// ══════════════════════════════════════════
function bindEvents() {
  // Sidebar toggle
  const sidebar = document.getElementById('sidebar');
  document.getElementById('sidebarToggle').onclick = () => sidebar.classList.add('collapsed');
  document.getElementById('sidebarOpenBtnAlt').onclick = () => sidebar.classList.remove('collapsed');

  // Section collapse
  document.getElementById('convSectionHeader').onclick = function() {
    this.classList.toggle('collapsed');
    document.getElementById('conversationList').classList.toggle('collapsed');
  };
  document.getElementById('algoSectionHeader').onclick = function() {
    this.classList.toggle('collapsed');
    document.getElementById('algoFilterTags').classList.toggle('collapsed');
    document.getElementById('algorithmList').classList.toggle('collapsed');
  };

  // New chat
  document.getElementById('newChatBtn').onclick = createNewConversation;

  // Search
  document.getElementById('sidebarSearch').oninput = (e) => {
    renderAlgorithmList(e.target.value.trim());
  };

  // Theme toggle
  document.getElementById('btnThemeToggle').onclick = toggleTheme;

  // Settings modal
  document.getElementById('btnSettings').onclick = () => {
    document.getElementById('settingsModal').style.display = '';
    document.getElementById('settingApiBase').value = API.baseUrl;
    document.getElementById('settingLlmUrl').value = API.llmConfig.url;
    document.getElementById('settingLlmKey').value = API.llmConfig.key;
    document.getElementById('settingLlmModel').value = API.llmConfig.model;
    document.getElementById('settingParticles').value = localStorage.getItem('algochat_particles') !== 'false' ? 'true' : 'false';
  };
  document.getElementById('settingsModalClose').onclick = () => {
    document.getElementById('settingsModal').style.display = 'none';
  };
  document.getElementById('settingsModal').onclick = (e) => {
    if (e.target.id === 'settingsModal') document.getElementById('settingsModal').style.display = 'none';
  };
  document.getElementById('settingApiBase').onchange = (e) => {
    API.baseUrl = e.target.value;
    localStorage.setItem('algochat_api_base', e.target.value);
  };
  document.getElementById('settingLlmUrl').onchange = (e) => {
    API.llmConfig.url = e.target.value;
    localStorage.setItem('algochat_llm_url', e.target.value);
  };
  document.getElementById('settingLlmKey').onchange = (e) => {
    API.llmConfig.key = e.target.value;
    localStorage.setItem('algochat_llm_key', e.target.value);
  };
  document.getElementById('settingLlmModel').onchange = (e) => {
    API.llmConfig.model = e.target.value;
    localStorage.setItem('algochat_llm_model', e.target.value);
  };
  document.getElementById('settingParticles').onchange = (e) => {
    localStorage.setItem('algochat_particles', e.target.value);
  };

  // Inline algo selector
  document.getElementById('btnAlgoInline').onclick = toggleAlgoSelector;
  document.getElementById('algoSelectorClose').onclick = closeAlgoSelector;
  document.getElementById('algoSelectorDropdown').onchange = (e) => {
    selectAlgorithm(e.target.value);
  };

  // Params toggle
  document.getElementById('algoParamsHeader').onclick = () => {
    document.getElementById('algoParamsBar').classList.toggle('collapsed');
  };

  // File upload
  document.getElementById('btnAttach').onclick = () => document.getElementById('fileInput').click();
  document.getElementById('fileInput').onchange = handleFileSelect;

  // File upload zone controls
  document.getElementById('fileUploadClearBtn').onclick = clearAllFiles;
  document.getElementById('fileUploadGrid').onclick = handleFileGridClick;

  // Text input
  const textarea = document.getElementById('messageInput');
  textarea.oninput = function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    document.getElementById('btnSend').disabled = !this.value.trim() && App.selectedFiles.length === 0;
  };
  textarea.onkeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };
  document.getElementById('btnSend').onclick = handleSend;

  // Preview panel
  document.getElementById('btnPreviewToggle').onclick = togglePreviewPanel;
  document.getElementById('btnPreviewClose').onclick = closePreviewPanel;
  document.getElementById('btnPreviewFullscreen').onclick = () => {
    const content = document.getElementById('previewContent');
    if (content.requestFullscreen) content.requestFullscreen();
  };
  document.getElementById('btnPreviewDownload').onclick = downloadPreviewContent;

  // Drag and drop
  const chatPanel = document.getElementById('chatPanel');
  chatPanel.ondragover = (e) => { e.preventDefault(); chatPanel.classList.add('drag-over'); };
  chatPanel.ondragleave = () => chatPanel.classList.remove('drag-over');
  chatPanel.ondrop = (e) => {
    e.preventDefault();
    chatPanel.classList.remove('drag-over');
    addFiles(e.dataTransfer.files);
  };
}

// ══════════════════════════════════════════
// THEME
// ══════════════════════════════════════════
function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('algochat_theme', next);
  updateThemeLabel(next);
  // Redraw charts with new theme
  Object.values(App.chartInstances).forEach(c => {
    c.options.scales && Object.values(c.options.scales).forEach(s => {
      if (s.grid) s.grid.color = getComputedStyle(document.documentElement).getPropertyValue('--chart-grid').trim();
      if (s.ticks) s.ticks.color = getComputedStyle(document.documentElement).getPropertyValue('--chart-tick').trim();
    });
    c.update('none');
  });
}
function updateThemeLabel(theme) {
  document.getElementById('themeLabel').textContent = theme === 'dark' ? '浅色模式' : '深色模式';
}

// ══════════════════════════════════════════
// ALGORITHM LIST
// ══════════════════════════════════════════
function renderAlgoFilterTags() {
  const container = document.getElementById('algoFilterTags');
  container.innerHTML = `<span class="algo-filter-tag ${App.activeCategory === null ? 'active' : ''}" data-cat="">全部</span>` +
    App.algoCategories.map(c =>
      `<span class="algo-filter-tag ${App.activeCategory === c ? 'active' : ''}" data-cat="${c}">${c}</span>`
    ).join('');
  container.querySelectorAll('.algo-filter-tag').forEach(tag => {
    tag.onclick = () => {
      App.activeCategory = tag.dataset.cat || null;
      renderAlgoFilterTags();
      renderAlgorithmList();
    };
  });
}

function renderAlgorithmList(search = '') {
  const container = document.getElementById('algorithmList');
  let algos = App.algorithms;
  if (App.activeCategory) algos = algos.filter(a => a.category === App.activeCategory);
  if (search) algos = algos.filter(a => a.name.includes(search) || a.description.includes(search) || a.id.includes(search.toLowerCase()));

  container.innerHTML = algos.map(a => `
    <div class="algo-item ${App.selectedAlgo?.id === a.id ? 'active' : ''}" data-algo-id="${a.id}">
      <span class="algo-item-icon">${a.icon}</span>
      <div class="algo-item-info">
        <div class="algo-item-name">${a.name}</div>
        <div class="algo-item-desc">${a.description}</div>
      </div>
      <span class="algo-item-badge">${a.category}</span>
    </div>
  `).join('');

  container.querySelectorAll('.algo-item').forEach(el => {
    el.onclick = () => selectAlgorithm(el.dataset.algoId);
  });
}

function renderWelcomeHints() {
  const container = document.getElementById('welcomeHints');
  const hints = App.algorithms.slice(0, 4).map(a =>
    `<div class="hint-card" data-algo-id="${a.id}"><span class="hint-icon">${a.icon}</span>${a.name}</div>`
  ).join('');
  container.innerHTML = hints;
  container.querySelectorAll('.hint-card').forEach(el => {
    el.onclick = () => {
      selectAlgorithm(el.dataset.algoId);
      document.getElementById('messageInput').focus();
    };
  });
}

// ══════════════════════════════════════════
// ALGORITHM SELECTION & PARAMS
// ══════════════════════════════════════════
function populateAlgoSelector() {
  const select = document.getElementById('algoSelectorDropdown');
  select.innerHTML = '<option value="">— 选择算法 —</option>' +
    App.algorithms.map(a => `<option value="${a.id}">${a.icon} ${a.name}</option>`).join('');
}

function selectAlgorithm(algoId) {
  const algo = App.algorithms.find(a => a.id === algoId);
  if (!algo) return;
  App.selectedAlgo = algo;

  // Update sidebar
  renderAlgorithmList();

  // Show inline selector
  document.getElementById('algoSelectorBar').style.display = '';
  document.getElementById('algoSelectorDropdown').value = algoId;

  // Update header
  document.getElementById('btnAlgoInline').classList.add('active');

  // Render params
  renderAlgoParams(algo);

  // Update chat header
  document.getElementById('chatSubInfo').textContent = `当前算法: ${algo.icon} ${algo.name}`;

  // Show file upload zone if not visible and there are files
  if (App.selectedFiles.length > 0) {
    document.getElementById('fileUploadZone').style.display = '';
  }
}

function toggleAlgoSelector() {
  const bar = document.getElementById('algoSelectorBar');
  if (bar.style.display === 'none') {
    bar.style.display = '';
    document.getElementById('btnAlgoInline').classList.add('active');
  } else {
    closeAlgoSelector();
  }
}

function closeAlgoSelector() {
  App.selectedAlgo = null;
  document.getElementById('algoSelectorBar').style.display = 'none';
  document.getElementById('algoParamsBar').style.display = 'none';
  document.getElementById('btnAlgoInline').classList.remove('active');
  document.getElementById('chatSubInfo').textContent = '选择算法或输入消息开始';
  renderAlgorithmList();
}

function renderAlgoParams(algo) {
  const bar = document.getElementById('algoParamsBar');
  const fields = document.getElementById('algoParamsFields');
  const keys = Object.keys(algo.params || {});

  if (keys.length === 0) {
    bar.style.display = 'none';
    return;
  }

  bar.style.display = '';
  bar.classList.remove('collapsed');

  fields.innerHTML = keys.map(key => {
    const p = algo.params[key];
    if (p.type === 'int' || p.type === 'float') {
      const min = p.min ?? 0;
      const max = p.max ?? 100;
      const step = p.step || (p.type === 'float' ? 0.1 : 1);
      const val = p.default;
      return `<div class="algo-param-input">
        <label>${p.label || key}</label>
        <input type="range" min="${min}" max="${max}" step="${step}" value="${val}" data-param="${key}" data-type="${p.type}"
          oninput="this.nextElementSibling.textContent=this.value" />
        <span class="algo-param-value">${val}</span>
      </div>`;
    }
    if (p.type === 'bool' || p.type === 'boolean') {
      return `<div class="algo-param-input">
        <label>${p.label || key}</label>
        <select data-param="${key}" data-type="bool">
          <option value="true" ${p.default === true ? 'selected' : ''}>是</option>
          <option value="false" ${p.default === false ? 'selected' : ''}>否</option>
        </select>
      </div>`;
    }
    if (p.enum || p.options) {
      const opts = p.enum || p.options;
      return `<div class="algo-param-input">
        <label>${p.label || key}</label>
        <select data-param="${key}" data-type="string">
          ${opts.map(opt => {
            const val = typeof opt === 'object' ? opt.value : opt;
            const label = typeof opt === 'object' ? opt.label : opt;
            return `<option value="${val}" ${p.default === val ? 'selected' : ''}>${label}</option>`;
          }).join('')}
        </select>
      </div>`;
    }
    // 根据参数名智能推断下拉选项
    const lowerKey = key.toLowerCase();
    const lowerLabel = (p.label || key).toLowerCase();
    if (lowerKey.includes('mode') || lowerKey.includes('type') || 
        lowerLabel.includes('模式') || lowerLabel.includes('类型')) {
      const defaultOpts = [
        { value: 'standard', label: '标准' },
        { value: 'fast', label: '快速' },
        { value: 'precise', label: '精确' }
      ];
      return `<div class="algo-param-input">
        <label>${p.label || key}</label>
        <select data-param="${key}" data-type="string">
          ${defaultOpts.map(opt => `<option value="${opt.value}" ${p.default === opt.value ? 'selected' : ''}>${opt.label}</option>`).join('')}
        </select>
      </div>`;
    }
    return `<div class="algo-param-input">
      <label>${p.label || key}</label>
      <select data-param="${key}" data-type="string">
        <option value="${p.default ?? ''}" selected>${p.default ?? '— 选择 —'}</option>
      </select>
    </div>`;
  }).join('');
}

function getAlgoParams() {
  const params = {};
  document.querySelectorAll('#algoParamsFields [data-param]').forEach(el => {
    const key = el.dataset.param;
    const type = el.dataset.type;
    params[key] = type === 'int' ? parseInt(el.value) : type === 'float' ? parseFloat(el.value) : el.value;
  });
  return params;
}

// ══════════════════════════════════════════
// FILE UPLOAD
// ══════════════════════════════════════════
function handleFileSelect(e) {
  addFiles(e.target.files);
  e.target.value = '';
}

function addFiles(fileList) {
  const newFiles = Array.from(fileList);
  newFiles.forEach(f => {
    App.selectedFiles.push(f);
    App.selectedFileIndices.push(App.selectedFiles.length - 1);
  });
  renderFileUploadZone();
}

function renderFileUploadZone() {
  const zone = document.getElementById('fileUploadZone');
  const grid = document.getElementById('fileUploadGrid');
  const count = document.getElementById('fileUploadCount');

  if (App.selectedFiles.length === 0) {
    zone.style.display = 'none';
    return;
  }

  zone.style.display = '';
  count.textContent = App.selectedFiles.length;

  grid.innerHTML = App.selectedFiles.map((f, i) => {
    const icon = getFileIcon(f.name);
    const ext = getFileExt(f.name);
    const size = formatFileSize(f.size);
    const selected = App.selectedFileIndices.includes(i);
    return `<div class="file-grid-item ${selected ? 'selected' : ''}" data-idx="${i}">
      <span class="file-checkbox">✓</span>
      <button class="file-grid-item-remove" data-remove="${i}" title="移除">✕</button>
      <span class="file-grid-item-icon">${icon}</span>
      <span class="file-grid-item-name" title="${f.name}">${f.name}</span>
      <span class="file-grid-item-ext">${ext}</span>
      <span class="file-grid-item-size">${size}</span>
    </div>`;
  }).join('') +
  `<button class="file-grid-add" id="fileGridAddBtn">
    <span class="file-grid-add-icon">+</span>添加
  </button>`;

  // Bind add button
  const addBtn = document.getElementById('fileGridAddBtn');
  if (addBtn) addBtn.onclick = () => document.getElementById('fileInput').click();

  // Update send button
  document.getElementById('btnSend').disabled = false;
}

function handleFileGridClick(e) {
  // Remove button
  const removeBtn = e.target.closest('[data-remove]');
  if (removeBtn) {
    const idx = parseInt(removeBtn.dataset.remove);
    App.selectedFiles.splice(idx, 1);
    App.selectedFileIndices = App.selectedFileIndices
      .filter(i => i !== idx)
      .map(i => i > idx ? i - 1 : i);
    renderFileUploadZone();
    return;
  }

  // Toggle select
  const item = e.target.closest('.file-grid-item');
  if (item) {
    const idx = parseInt(item.dataset.idx);
    if (App.selectedFileIndices.includes(idx)) {
      App.selectedFileIndices = App.selectedFileIndices.filter(i => i !== idx);
    } else {
      App.selectedFileIndices.push(idx);
    }
    renderFileUploadZone();
  }
}

function clearAllFiles() {
  App.selectedFiles = [];
  App.selectedFileIndices = [];
  renderFileUploadZone();
}

function getFileIcon(name) {
  const ext = name.split('.').pop().toLowerCase();
  const map = { csv:'📊', xlsx:'📊', xls:'📊', json:'📋', txt:'📄', pdf:'📕', png:'🖼', jpg:'🖼', jpeg:'🖼', svg:'🖼', gif:'🖼' };
  return map[ext] || '📎';
}

function getFileExt(name) {
  return '.' + name.split('.').pop().toLowerCase();
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + 'B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + 'KB';
  return (bytes / 1048576).toFixed(1) + 'MB';
}

// ══════════════════════════════════════════
// CONVERSATIONS
// ══════════════════════════════════════════
function createNewConversation() {
  const conv = {
    id: 'conv_' + Date.now(),
    title: '新对话',
    messages: [],
    createdAt: new Date().toISOString(),
  };
  App.conversations.unshift(conv);
  App.currentConvId = conv.id;
  renderConversationList();
  clearChat();
  document.getElementById('chatTitle').textContent = '新对话';
  document.getElementById('chatSubInfo').textContent = '选择算法或输入消息开始';
  saveConversations();
}

function loadConversations() {
  try {
    App.conversations = JSON.parse(localStorage.getItem('algochat_conversations') || '[]');
  } catch { App.conversations = []; }
  if (App.conversations.length === 0) createNewConversation();
  else App.currentConvId = App.conversations[0].id;
}

function saveConversations() {
  try {
    // Don't save chart data in messages to avoid huge localStorage
    const lite = App.conversations.map(c => ({
      ...c,
      messages: c.messages.map(m => ({ ...m, results: m.results?.map(r => ({ ...r, data: r.type === 'chart' ? null : r.data })) }))
    }));
    localStorage.setItem('algochat_conversations', JSON.stringify(lite));
  } catch {}
}

function renderConversationList() {
  const container = document.getElementById('conversationList');
  container.innerHTML = App.conversations.map(c => `
    <div class="conv-item ${c.id === App.currentConvId ? 'active' : ''}" data-conv-id="${c.id}">
      <span class="conv-item-icon">💬</span>
      <div class="conv-item-info">
        <div class="conv-item-title">${c.title}</div>
        <div class="conv-item-sub">${new Date(c.createdAt).toLocaleDateString('zh-CN')}</div>
      </div>
      <button class="conv-item-delete" data-delete-conv="${c.id}" title="删除">✕</button>
    </div>
  `).join('');

  container.querySelectorAll('.conv-item').forEach(el => {
    el.onclick = (e) => {
      if (e.target.closest('[data-delete-conv]')) return;
      switchConversation(el.dataset.convId);
    };
  });
  container.querySelectorAll('[data-delete-conv]').forEach(btn => {
    btn.onclick = (e) => {
      e.stopPropagation();
      deleteConversation(btn.dataset.deleteConv);
    };
  });
}

function switchConversation(convId) {
  App.currentConvId = convId;
  const conv = App.conversations.find(c => c.id === convId);
  if (!conv) return;
  renderConversationList();
  renderMessages(conv);
  document.getElementById('chatTitle').textContent = conv.title;
}

function deleteConversation(convId) {
  App.conversations = App.conversations.filter(c => c.id !== convId);
  if (App.currentConvId === convId) {
    if (App.conversations.length > 0) switchConversation(App.conversations[0].id);
    else createNewConversation();
  }
  renderConversationList();
  saveConversations();
}

function clearChat() {
  document.getElementById('messages').innerHTML = '';
  document.getElementById('welcomeScreen').classList.remove('hidden');
  // Destroy old charts
  Object.values(App.chartInstances).forEach(c => c.destroy());
  App.chartInstances = {};
}

// ══════════════════════════════════════════
// SEND MESSAGE
// ══════════════════════════════════════════
async function handleSend() {
  if (App.isProcessing) return;
  const input = document.getElementById('messageInput');
  const text = input.value.trim();
  const files = App.selectedFileIndices.map(i => App.selectedFiles[i]).filter(Boolean);

  if (!text && files.length === 0) return;

  const conv = App.conversations.find(c => c.id === App.currentConvId);
  if (!conv) return;

  // Hide welcome
  document.getElementById('welcomeScreen').classList.add('hidden');

  // Add user message
  const userMsg = { role: 'user', content: text, files: files.map(f => ({ name: f.name, size: f.size })) };
  conv.messages.push(userMsg);
  renderUserMessage(userMsg);

  // Update conv title
  if (conv.messages.filter(m => m.role === 'user').length === 1) {
    conv.title = text || (files[0]?.name || '文件处理');
    document.getElementById('chatTitle').textContent = conv.title;
    renderConversationList();
  }

  // Clear input
  input.value = '';
  input.style.height = 'auto';
  document.getElementById('btnSend').disabled = true;

  // Show typing
  const typingEl = showTypingIndicator();

  App.isProcessing = true;

  // 判断使用流式LLM对话还是普通请求
  const useLLMStream = !App.selectedAlgo && (text || files.length === 0);

  if (useLLMStream) {
    // 使用流式LLM对话
    typingEl.remove();
    const assistantMsg = {
      role: 'assistant',
      content: '',
      results: [],
      error: false,
    };
    conv.messages.push(assistantMsg);
    
    // 创建消息元素用于流式更新
    const msgEl = renderStreamAssistantMessage(assistantMsg);
    
    await API.sendLLMMessageStream(
      conv.id,
      text,
      (accumulated) => {
        // onChunk - 更新消息内容
        assistantMsg.content = accumulated;
        updateStreamMessage(msgEl, accumulated);
      },
      (response) => {
        // onComplete
        App.isProcessing = false;
        saveConversations();
      },
      (error) => {
        // onError
        App.isProcessing = false;
        assistantMsg.content = '对话失败: ' + error;
        assistantMsg.error = true;
        updateStreamMessage(msgEl, assistantMsg.content, true);
        saveConversations();
      }
    );

    // Clear files after send
    App.selectedFiles = [];
    App.selectedFileIndices = [];
    renderFileUploadZone();
    
    return;
  }

  try {
    let response;
    if (App.selectedAlgo && files.length > 0) {
      // Run algorithm
      const params = getAlgoParams();
      response = await API.runAlgorithm(App.selectedAlgo.id, files, params);
    } else {
      // Chat (non-streaming fallback)
      response = await API.sendMessage(conv.id, text, files);
    }

    // Remove typing
    typingEl.remove();

    // Add assistant message
    const assistantMsg = {
      role: 'assistant',
      content: response.message || '',
      results: response.results || [],
      error: response.type === 'error',
    };
    conv.messages.push(assistantMsg);
    renderAssistantMessage(assistantMsg);

    // Show preview toggle if results exist
    if (response.results?.length) {
      document.getElementById('btnPreviewToggle').style.display = '';
    }
  } catch (e) {
    typingEl.remove();
    renderAssistantMessage({ role: 'assistant', content: '处理失败: ' + e.message, error: true });
  }

  App.isProcessing = false;

  // Clear files after send
  App.selectedFiles = [];
  App.selectedFileIndices = [];
  renderFileUploadZone();

  saveConversations();
}

// 渲染流式消息元素
function renderStreamAssistantMessage(msg) {
  const el = document.createElement('div');
  el.className = 'message assistant streaming';

  el.innerHTML = `
    <div class="message-avatar">◈</div>
    <div class="message-content">
      <div class="message-bubble"><span class="stream-content"></span><span class="stream-cursor">▋</span></div>
    </div>
  `;
  document.getElementById('messages').appendChild(el);
  scrollToBottom();
  return el;
}

// 更新流式消息内容
function updateStreamMessage(el, content, isError = false) {
  const bubble = el.querySelector('.message-bubble');
  const contentSpan = el.querySelector('.stream-content');
  
  if (isError) {
    bubble.style.color = 'var(--color-red-light)';
  }
  
  contentSpan.innerHTML = renderMarkdown(content);
  scrollToBottom();
}

// ══════════════════════════════════════════
// RENDER MESSAGES
// ══════════════════════════════════════════
function renderMessages(conv) {
  clearChat();
  if (conv.messages.length === 0) {
    document.getElementById('welcomeScreen').classList.remove('hidden');
    return;
  }
  document.getElementById('welcomeScreen').classList.add('hidden');
  conv.messages.forEach(msg => {
    if (msg.role === 'user') renderUserMessage(msg);
    else renderAssistantMessage(msg);
  });
}

function renderUserMessage(msg) {
  document.getElementById('welcomeScreen').classList.add('hidden');
  const el = document.createElement('div');
  el.className = 'message user';

  let filesHtml = '';
  if (msg.files?.length) {
    filesHtml = `<div class="file-upload-message">${msg.files.map(f =>
      `<div class="file-card"><span class="file-card-icon">${getFileIcon(f.name)}</span><span class="file-card-name">${f.name}</span><span class="file-card-size">${formatFileSize(f.size)}</span></div>`
    ).join('')}</div>`;
  }

  el.innerHTML = `
    <div class="message-avatar">你</div>
    <div class="message-content">
      ${filesHtml}
      ${msg.content ? `<div class="message-bubble">${escapeHtml(msg.content)}</div>` : ''}
    </div>
  `;
  document.getElementById('messages').appendChild(el);
  scrollToBottom();
}

function renderAssistantMessage(msg) {
  const el = document.createElement('div');
  el.className = 'message assistant';

  let contentHtml = '';
  if (msg.error) {
    contentHtml = `<div class="message-bubble" style="color:var(--color-red-light)">❌ ${escapeHtml(msg.content)}</div>`;
  } else if (msg.results?.length) {
    // Message text
    contentHtml = `<div class="message-bubble">${renderMarkdown(msg.content)}</div>`;
    // Workflow node
    contentHtml += renderWorkflowNode(msg.results);
  } else {
    contentHtml = `<div class="message-bubble">${renderMarkdown(msg.content)}</div>`;
  }

  el.innerHTML = `
    <div class="message-avatar">◈</div>
    <div class="message-content">${contentHtml}</div>
  `;
  document.getElementById('messages').appendChild(el);

  // Render charts after DOM insert — only for currently visible canvases
  if (msg.results?.length) {
    setTimeout(() => {
      const messagesEl = document.getElementById('messages');
      const lastNode = messagesEl.querySelector('.workflow-node:last-child');
      if (lastNode) {
        lastNode.querySelectorAll('.result-panel-content').forEach(contentEl => {
          if (contentEl.style.display !== 'none') {
            const canvas = contentEl.querySelector('canvas');
            if (canvas && !App.chartInstances[canvas.id]) {
              const idx = parseInt(contentEl.dataset.resultIdx);
              if (msg.results[idx]?.type === 'chart') {
                renderChartAt(msg.results[idx], canvas.id);
              }
            }
          }
        });
      }
    }, 80);
  }

  scrollToBottom();
}

function showTypingIndicator() {
  const el = document.createElement('div');
  el.className = 'message assistant';
  el.id = 'typingIndicator';
  el.innerHTML = `
    <div class="message-avatar">◈</div>
    <div class="message-content">
      <div class="typing-indicator"><span></span><span></span><span></span></div>
    </div>
  `;
  document.getElementById('messages').appendChild(el);
  scrollToBottom();
  return el;
}

// ══════════════════════════════════════════
// WORKFLOW NODE
// ══════════════════════════════════════════
function renderWorkflowNode(results) {
  const nodeId = 'wn_' + Date.now();

  // ── Analyze groups ──
  const groupOrder = [];
  const groupMap = {};   // group name → [result indices]
  const ungrouped = [];  // [result indices] without group

  results.forEach((r, i) => {
    if (r.group) {
      if (!groupMap[r.group]) {
        groupOrder.push(r.group);
        groupMap[r.group] = [];
      }
      groupMap[r.group].push(i);
    } else {
      ungrouped.push(i);
    }
  });

  const hasGroups = groupOrder.length > 0;

  // ── Build flat index map for result-idx ──
  // We need a stable index for each result to reference in tab clicks
  // results[i] always has result-idx = i

  // ── Build content panels (all results, always present) ──
  const allContentsHtml = results.map((r, i) => {
    const contentId = `${nodeId}_${i}`;
    let inner = '';
    switch (r.type) {
      case 'chart':
        inner = `<div class="preview-chart-container"><canvas id="chart_${contentId}"></canvas></div>`;
        break;
      case 'table':
        inner = renderTableHTML(r);
        break;
      case 'image':
        inner = `<div style="text-align:center"><img src="${r.src}" alt="${escapeHtml(r.name)}" class="result-image" style="max-height:400px" /></div>`;
        break;
      case 'document':
        inner = `<div class="preview-doc markdown-body">${renderMarkdown(r.content || '')}</div>`;
        break;
      default:
        inner = `<pre class="preview-doc">${escapeHtml(JSON.stringify(r, null, 2))}</pre>`;
    }
    const isVisible = hasGroups ? (i === (ungrouped.length > 0 ? ungrouped[0] : groupMap[groupOrder[0]][0])) : (i === 0);
    return `<div class="result-panel-content" id="content_${contentId}" data-result-idx="${i}" style="${isVisible ? '' : 'display:none'}">${inner}</div>`;
  }).join('');

  let tabsAreaHtml = '';

  if (hasGroups) {
    // ── GROUPED LAYOUT: two-level tabs ──
    // Level 1: group tabs + ungrouped tabs
    let firstActiveIdx = ungrouped.length > 0 ? ungrouped[0] : groupMap[groupOrder[0]][0];

    const l1Tabs = [];
    // Ungrouped items first (top-level tabs)
    ungrouped.forEach(idx => {
      const r = results[idx];
      const typeIcon = { chart: '📈', table: '📊', image: '🖼', document: '📄' }[r.type] || '📎';
      l1Tabs.push(`<button class="group-tab${idx === firstActiveIdx ? ' active' : ''}" data-group="" data-result-idx="${idx}" data-tab="${nodeId}_${idx}">
        <span class="group-tab-icon">${typeIcon}</span>
        <span class="group-tab-label">${escapeHtml(r.name || r.type)}</span>
      </button>`);
    });
    // Group tabs
    groupOrder.forEach(g => {
      const count = groupMap[g].length;
      l1Tabs.push(`<button class="group-tab${groupMap[g].includes(firstActiveIdx) ? ' active' : ''}" data-group="${escapeAttr(g)}">
        <span class="group-tab-icon">📁</span>
        <span class="group-tab-label">${escapeHtml(g)}</span>
        <span class="group-tab-count">${count}</span>
      </button>`);
    });

    // Level 2: sub-tabs per group (hidden by default)
    let l2Html = '';
    groupOrder.forEach(g => {
      const subTabs = groupMap[g].map(idx => {
        const r = results[idx];
        const typeIcon = { chart: '📈', table: '📊', image: '🖼', document: '📄' }[r.type] || '📎';
        const isActive = idx === firstActiveIdx ? 'active' : '';
        return `<button class="result-tab ${isActive}" data-tab="${nodeId}_${idx}" data-result-idx="${idx}">
          <span class="result-tab-icon">${typeIcon}</span>
          <span class="result-tab-label">${escapeHtml(r.name || r.type)}</span>
          <span class="result-tab-type ${r.type}">${r.type}</span>
        </button>`;
      }).join('');
      const showSubtabs = groupMap[g].includes(firstActiveIdx);
      l2Html += `<div class="result-panel-subtabs" data-subgroup="${escapeAttr(g)}" style="${showSubtabs ? '' : 'display:none'}">${subTabs}</div>`;
    });

    tabsAreaHtml = `
      <div class="group-tabs-row">${l1Tabs.join('')}</div>
      ${l2Html}
    `;
  } else {
    // ── FLAT LAYOUT: original single-level tabs ──
    const tabsHtml = results.map((r, i) => {
      const typeIcon = { chart: '📈', table: '📊', image: '🖼', document: '📄' }[r.type] || '📎';
      const typeClass = r.type;
      return `<button class="result-tab ${i === 0 ? 'active' : ''}" data-tab="${nodeId}_${i}" data-result-idx="${i}">
        <span class="result-tab-icon">${typeIcon}</span>
        <span class="result-tab-label">${escapeHtml(r.name || r.type)}</span>
        <span class="result-tab-type ${typeClass}">${r.type}</span>
      </button>`;
    }).join('');
    tabsAreaHtml = `<div class="result-panel-tabs">${tabsHtml}</div>`;
  }

  // Build action buttons
  const actionsHtml = `
    <button class="result-action-btn download" onclick="downloadCurrentResult(this)" title="下载">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
      下载
    </button>
    <button class="result-action-btn reuse" onclick="previewResultInPanel(this)" title="在预览面板查看">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
      预览
    </button>
  `;

  return `
    <div class="workflow-node${hasGroups ? ' has-groups' : ''}" data-node-id="${nodeId}" data-results='${escapeAttr(JSON.stringify(results.map(r => ({...r, data: r.type === 'chart' ? null : r.data, src: r.type === 'image' ? '' : r.src}))))}'>
      <div class="workflow-node-header">
        <span class="workflow-node-icon">${App.selectedAlgo?.icon || '🧮'}</span>
        <span class="workflow-node-title">${App.selectedAlgo?.name || '算法结果'}</span>
        <span class="workflow-node-status done">✓ 完成</span>
        <span class="workflow-node-meta">${results.length} 个结果${hasGroups ? ' · ' + groupOrder.length + ' 个分组' : ''}</span>
      </div>
      <div class="result-panel">
        ${tabsAreaHtml}
        ${allContentsHtml}
        <div class="result-panel-actions">${actionsHtml}</div>
      </div>
    </div>
  `;
}

// Tab switching (handles both flat and grouped modes)
document.addEventListener('click', (e) => {
  // ── Group tab click (level 1) ──
  const groupTab = e.target.closest('.group-tab');
  if (groupTab) {
    const node = groupTab.closest('.workflow-node');
    if (!node) return;

    // Deactivate all group tabs
    node.querySelectorAll('.group-tab').forEach(t => t.classList.remove('active'));
    groupTab.classList.add('active');

    const groupName = groupTab.dataset.group;

    if (groupName === '') {
      // Ungrouped item — directly show its content
      const idx = parseInt(groupTab.dataset.resultIdx);
      node.querySelectorAll('.result-panel-content').forEach(c => c.style.display = 'none');
      const contentEl = node.querySelector(`.result-panel-content[data-result-idx="${idx}"]`);
      if (contentEl) contentEl.style.display = '';
      // Hide all sub-tabs
      node.querySelectorAll('.result-panel-subtabs').forEach(s => s.style.display = 'none');
      // Deactivate sub-tab active states
      node.querySelectorAll('.result-tab').forEach(t => t.classList.remove('active'));
      // Track active index
      node.dataset.activeIdx = idx;
      // Lazy render chart
      lazyRenderChart(node, idx);
    } else {
      // Group tab — show sub-tabs for this group, hide all content
      node.querySelectorAll('.result-panel-subtabs').forEach(s => {
        s.style.display = s.dataset.subgroup === groupName ? '' : 'none';
      });
      // Activate first sub-tab
      const subtabs = node.querySelectorAll(`.result-panel-subtabs[data-subgroup="${CSS.escape(groupName)}"] .result-tab`);
      if (subtabs.length > 0) {
        subtabs.forEach(t => t.classList.remove('active'));
        subtabs[0].classList.add('active');
        const idx = parseInt(subtabs[0].dataset.resultIdx);
        node.querySelectorAll('.result-panel-content').forEach(c => c.style.display = 'none');
        const contentEl = node.querySelector(`.result-panel-content[data-result-idx="${idx}"]`);
        if (contentEl) contentEl.style.display = '';
        lazyRenderChart(node, idx);
      }
    }
    return;
  }

  // ── Result tab click (level 2 in grouped, or flat mode) ──
  const tab = e.target.closest('.result-tab');
  if (!tab) return;
  const node = tab.closest('.workflow-node');
  if (!node) return;

  // Deactivate all result tabs in the same subtab row
  const subtabRow = tab.closest('.result-panel-subtabs') || tab.closest('.result-panel-tabs');
  if (subtabRow) {
    subtabRow.querySelectorAll('.result-tab').forEach(t => t.classList.remove('active'));
  }
  tab.classList.add('active');

  // Hide all content, show selected
  const idx = parseInt(tab.dataset.resultIdx);
  node.querySelectorAll('.result-panel-content').forEach(c => c.style.display = 'none');
  const contentEl = node.querySelector(`.result-panel-content[data-result-idx="${idx}"]`);
  if (contentEl) contentEl.style.display = '';

  // Lazy render chart
  lazyRenderChart(node, idx);
});

function lazyRenderChart(node, idx) {
  const conv = App.conversations.find(c => c.id === App.currentConvId);
  if (!conv) return;
  const lastAssistant = [...conv.messages].reverse().find(m => m.role === 'assistant' && m.results?.length);
  if (!lastAssistant || !lastAssistant.results[idx]) return;

  const result = lastAssistant.results[idx];
  if (result.type !== 'chart') return;

  const contentEl = node.querySelector(`.result-panel-content[data-result-idx="${idx}"]`);
  if (!contentEl) return;
  const canvas = contentEl.querySelector('canvas');
  if (!canvas || App.chartInstances[canvas.id]) return;

  renderChartAt(result, canvas.id);
}

// ══════════════════════════════════════════
// CHART RENDERING
// ══════════════════════════════════════════
function renderChart(result) {
  // Find the canvas element
  const conv = App.conversations.find(c => c.id === App.currentConvId);
  if (!conv) return;
  // Find canvas in the last workflow node
  const messagesEl = document.getElementById('messages');
  const canvases = messagesEl.querySelectorAll(`canvas[id^="chart_"]`);
  // Find the one that matches this result
  for (const canvas of canvases) {
    const id = canvas.id;
    if (!App.chartInstances[id] && canvas.offsetParent !== null) {
      renderChartAt(result, id);
      return;
    }
  }
  // Try by constructing the expected ID
  const nodes = messagesEl.querySelectorAll('.workflow-node');
  for (const node of nodes) {
    const tabs = node.querySelectorAll('.result-tab');
    for (const tab of tabs) {
      if (tab.dataset.resultIdx !== undefined) {
        const idx = parseInt(tab.dataset.resultIdx);
        const tabId = tab.dataset.tab;
        const chartId = `chart_${tabId}`;
        const canvas = document.getElementById(chartId);
        if (canvas && !App.chartInstances[chartId]) {
          // Find matching result
          if (conv) {
            const lastAsst = [...conv.messages].reverse().find(m => m.role === 'assistant' && m.results?.length);
            if (lastAsst && lastAsst.results[idx]) {
              renderChartAt(lastAsst.results[idx], chartId);
            }
          }
        }
      }
    }
  }
}

function renderChartAt(result, canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || App.chartInstances[canvasId]) return;

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const gridColor = isDark ? 'rgba(125,158,141,0.1)' : 'rgba(0,0,0,0.06)';
  const tickColor = isDark ? '#96877B' : '#6B6560';
  const palette = ['#7D9E8D','#8E4E26','#992D1E','#A0B9AA','#C49B60','#0EA5E9','#D4A867','#D4594A'];

  const config = {
    type: result.chartType === 'pie' ? 'pie' : result.chartType === 'bar' ? 'bar' : result.chartType === 'scatter' ? 'scatter' : 'line',
    data: {
      labels: result.data?.labels,
      datasets: (result.data?.datasets || []).map((ds, i) => ({
        ...ds,
        backgroundColor: result.chartType === 'pie' ? palette : palette[i % palette.length] + (result.chartType === 'bar' ? '99' : '33'),
        borderColor: palette[i % palette.length],
        borderWidth: result.chartType === 'scatter' ? 0 : 2,
        pointRadius: result.chartType === 'scatter' ? 5 : 3,
        tension: 0.3,
        fill: result.chartType === 'line' ? false : undefined,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: tickColor, font: { family: 'Inter', size: 11 } } },
      },
      scales: result.chartType === 'pie' ? {} : {
        x: { grid: { color: gridColor }, ticks: { color: tickColor, font: { family: 'Inter', size: 10 } } },
        y: { grid: { color: gridColor }, ticks: { color: tickColor, font: { family: 'Inter', size: 10 } } },
      },
    },
  };

  App.chartInstances[canvasId] = new Chart(canvas, config);
}

// ══════════════════════════════════════════
// TABLE RENDERING
// ══════════════════════════════════════════
function renderTableHTML(result) {
  if (!result.columns || !result.rows) return '<p style="color:var(--color-text-muted)">无表格数据</p>';
  return `<div class="table-container" style="max-height:300px;overflow:auto">
    <table>
      <thead><tr>${result.columns.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead>
      <tbody>${result.rows.map(row => `<tr>${row.map(cell => `<td>${escapeHtml(String(cell))}</td>`).join('')}</tr>`).join('')}</tbody>
    </table>
  </div>`;
}

// ══════════════════════════════════════════
// PREVIEW PANEL
// ══════════════════════════════════════════
function togglePreviewPanel() {
  const panel = document.getElementById('previewPanel');
  panel.classList.toggle('open');
  document.getElementById('btnPreviewToggle').classList.toggle('active');
}

function closePreviewPanel() {
  document.getElementById('previewPanel').classList.remove('open');
  document.getElementById('btnPreviewToggle').classList.remove('active');
}

function previewResultInPanel(btn) {
  const node = btn.closest('.workflow-node');
  if (!node) return;

  // Find active result index — could be from a result-tab or from group-tab (stored in dataset)
  const activeTab = node.querySelector('.result-tab.active');
  let idx = activeTab ? parseInt(activeTab.dataset.resultIdx) : parseInt(node.dataset.activeIdx);
  if (isNaN(idx)) idx = 0;

  const conv = App.conversations.find(c => c.id === App.currentConvId);
  if (!conv) return;

  const lastAsst = [...conv.messages].reverse().find(m => m.role === 'assistant' && m.results?.length);
  if (!lastAsst) return;

  const result = lastAsst.results[idx];
  if (!result) return;

  const panel = document.getElementById('previewPanel');
  const content = document.getElementById('previewContent');
  const title = document.getElementById('previewTitle');

  title.textContent = result.name || '预览';

  // Destroy old preview chart
  if (App.previewChartInstance) { App.previewChartInstance.destroy(); App.previewChartInstance = null; }

  switch (result.type) {
    case 'chart':
      content.innerHTML = `<div class="preview-chart-container"><canvas id="previewChart"></canvas></div>`;
      setTimeout(() => {
        App.previewChartInstance = new Chart(document.getElementById('previewChart'), {
          type: result.chartType === 'pie' ? 'pie' : result.chartType === 'bar' ? 'bar' : result.chartType === 'scatter' ? 'scatter' : 'line',
          data: {
            labels: result.data?.labels,
            datasets: (result.data?.datasets || []).map((ds, i) => ({
              ...ds,
              backgroundColor: ['#7D9E8D','#8E4E26','#992D1E','#A0B9AA','#C49B60'][i % 5],
              borderColor: ['#7D9E8D','#8E4E26','#992D1E','#A0B9AA','#C49B60'][i % 5],
              borderWidth: 2,
            })),
          },
          options: { responsive: true, maintainAspectRatio: false },
        });
      }, 100);
      break;
    case 'table':
      content.innerHTML = `<div class="preview-table-container">${renderTableHTML(result)}</div>`;
      break;
    case 'image':
      content.innerHTML = `<div class="preview-image-container"><img src="${result.src}" alt="${escapeHtml(result.name)}" /></div>`;
      break;
    case 'document':
      content.innerHTML = `<pre class="preview-doc">${escapeHtml(result.content || '')}</pre>`;
      break;
    default:
      content.innerHTML = `<pre class="preview-doc">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
  }

  panel.classList.add('open');
  document.getElementById('btnPreviewToggle').classList.add('active');
  document.getElementById('btnPreviewToggle').style.display = '';

  // Store current preview result for download
  panel.dataset.currentResult = JSON.stringify({ ...result, data: result.type === 'chart' ? null : result.data, src: result.type === 'image' ? '' : result.src });
}

function downloadPreviewContent() {
  // Simple: find the active result in the active workflow node
  const panel = document.getElementById('previewPanel');
  const title = document.getElementById('previewTitle').textContent;
  const content = document.getElementById('previewContent');

  // Try to get table data
  const table = content.querySelector('table');
  if (table) {
    const wb = XLSX.utils.table_to_book(table);
    XLSX.writeFile(wb, title.replace(/\.\w+$/, '') + '.xlsx');
    return;
  }

  // Try to get image
  const img = content.querySelector('img');
  if (img && img.src) {
    const a = document.createElement('a');
    a.href = img.src;
    a.download = title || 'image.png';
    a.click();
    return;
  }

  // Try to get document text
  const doc = content.querySelector('.preview-doc');
  if (doc) {
    const blob = new Blob([doc.textContent], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = title || 'document.txt';
    a.click();
    URL.revokeObjectURL(a.href);
    return;
  }
}

function downloadCurrentResult(btn) {
  // Trigger download from the active result in the workflow node
  previewResultInPanel(btn);
  setTimeout(() => downloadPreviewContent(), 200);
}

// ══════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════
function escapeHtml(str) {
  if (!str) return '';
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function escapeAttr(str) {
  return escapeHtml(str);
}

function renderMarkdown(text) {
  if (!text) return '';
  if (typeof marked !== 'undefined') {
    try { return marked.parse(text); } catch {}
  }
  return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function scrollToBottom() {
  const el = document.getElementById('chatMessages');
  if (el) setTimeout(() => el.scrollTop = el.scrollHeight, 50);
}
