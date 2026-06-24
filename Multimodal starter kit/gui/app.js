/**
 * TGENAI Assistant Activity Monitor.
 * Vanilla JS websocket clients, state, and renderers.
 */

'use strict';

const TIMELINE_WS_URL = `ws://${location.host}/gui/ws`;
const DETAIL_WS_URL = `ws://${location.host}/gui/tool-details/ws`;
const RECONNECT_DELAY_MS = 3000;
const PING_INTERVAL_MS = 25000;
const MAX_DETAIL_MESSAGES = 80;

/** @typedef {{id?: string, timestamp?: number, event_type: string, run_id?: string, session_id?: string, payload?: Object}} GuiEvent */
/** @typedef {{id?: string, timestamp?: number, tool_name: string, message_type: string, title?: string, source?: string, payload?: Object, detail_key?: string}} ToolDetailMessage */
/** @typedef {{key: string, name: string, status: string, callId?: string, args?: unknown, resultPreview?: string, error?: string, durationMs?: number, images?: unknown[], detailKey?: string}} ToolCall */
/** @typedef {{id: string, message: string, status: string, content: string, timestamp: number, tools: ToolCall[]}} RunState */

class ManagedSocket {
  constructor(url, onMessage, onStatus) {
    this.url = url;
    this.onMessage = onMessage;
    this.onStatus = onStatus;
    this.socket = null;
    this.pingTimer = null;
    this.reconnectTimer = null;
  }

  connect() {
    this.socket = new WebSocket(this.url);

    this.socket.addEventListener('open', () => {
      this.onStatus(true);
      this.startPing();
    });

    this.socket.addEventListener('message', (event) => {
      const data = parseJson(event.data);
      if (!data || data.type === 'pong') return;
      this.onMessage(data);
    });

    this.socket.addEventListener('close', () => {
      this.onStatus(false);
      this.stopPing();
      this.scheduleReconnect();
    });

    this.socket.addEventListener('error', () => {
      this.socket?.close();
    });
  }

  startPing() {
    this.stopPing();
    this.pingTimer = window.setInterval(() => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify({ type: 'ping' }));
      }
    }, PING_INTERVAL_MS);
  }

  stopPing() {
    if (this.pingTimer) window.clearInterval(this.pingTimer);
    this.pingTimer = null;
  }

  scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, RECONNECT_DELAY_MS);
  }
}

class TimelineSocket extends ManagedSocket {
  constructor(app) {
    super(TIMELINE_WS_URL, (event) => app.handleTimelineEvent(event), (connected) => app.setTimelineStatus(connected));
  }
}

class ToolDetailSocket extends ManagedSocket {
  constructor(app) {
    super(DETAIL_WS_URL, (message) => app.handleToolDetail(message), (connected) => app.setDetailStatus(connected));
  }
}

class AppState {
  constructor() {
    this.runs = new Map();
    this.runOrder = [];
    this.activeRunId = null;
    this.activeToolKeysByLookup = new Map();
    this.toolKeysByCallId = new Map();
    this.detailsByToolCall = new Map();
    this.pendingDetailKeysByTool = new Map();
    this.openDetailKeysByTool = new Map();
    this.selectedToolKey = null;
    this.seenTimelineIds = new Set();
    this.seenDetailIds = new Set();
  }

  upsertRun(runId, message, timestamp) {
    const id = runId || `run-${timestamp || Date.now()}`;
    let run = this.runs.get(id);
    if (!run) {
      run = { id, message: '', status: 'running', content: '', timestamp: timestamp || Date.now() / 1000, tools: [] };
      this.runs.set(id, run);
      this.runOrder.unshift(id);
    }
    run.message = message || run.message;
    run.status = 'running';
    this.activeRunId = id;
    return run;
  }

  finishRun(runId, status, content) {
    const run = this.findRun(runId);
    if (!run) return;
    run.status = status;
    if (content) run.content = content;
    if (this.activeRunId === run.id) this.activeRunId = null;
  }

  addToken(runId, token) {
    const run = this.findRun(runId);
    if (run) run.content += token || '';
  }

  startTool(runId, toolName, args, eventId, toolCallId) {
    const run = this.findRun(runId) || this.upsertRun(runId, '', Date.now() / 1000);
    const key = this.buildToolKey(run.id, toolName, toolCallId || eventId || run.tools.length);
    const tool = { key, name: toolName || 'tool', callId: toolCallId || null, status: 'running', args };
    run.tools.push(tool);
    this.pushActiveToolKey(run.id, tool.name, key);
    if (tool.callId) this.toolKeysByCallId.set(this.callLookupKey(run.id, tool.callId), key);
    const pendingDetailKey = this.takePendingDetailKey(tool.name);
    if (pendingDetailKey) this.assignDetailKey(tool, pendingDetailKey);
    if (!this.selectedToolKey) this.selectedToolKey = tool.key;
    return tool;
  }

  finishTool(runId, toolName, status, payload) {
    const run = this.findRun(runId);
    const lookupRunId = run?.id || this.activeRunId || '';
    const toolCallId = payload?.tool_call_id || null;
    const toolKey = toolCallId
      ? this.toolKeysByCallId.get(this.callLookupKey(lookupRunId, toolCallId))
      : this.peekActiveToolKey(lookupRunId, toolName);
    const tool = this.findToolByKey(toolKey);
    if (!tool) return;
    tool.status = status;
    tool.resultPreview = payload?.result_preview || '';
    tool.error = payload?.error || '';
    tool.durationMs = payload?.duration_ms;
    tool.images = payload?.images || [];
    this.removeActiveToolKey(lookupRunId, tool.name, tool.key);
    if (tool.callId) this.toolKeysByCallId.delete(this.callLookupKey(lookupRunId, tool.callId));
  }

  addDetail(message) {
    const detailKey = this.resolveDetailKey(message);
    const messages = this.detailsByToolCall.get(detailKey) || [];
    message.detail_key = detailKey;
    messages.push(message);
    if (messages.length > MAX_DETAIL_MESSAGES) messages.shift();
    this.detailsByToolCall.set(detailKey, messages);
    if (!this.selectedToolKey) this.selectedToolKey = detailKey;
    return detailKey;
  }

  findToolByDetailKey(detailKey) {
    if (!detailKey) return null;
    for (const runId of this.runOrder) {
      const run = this.runs.get(runId);
      if (!run) continue;
      for (const tool of run.tools) {
        if (tool.key === detailKey || tool.detailKey === detailKey) return tool;
      }
    }
    return null;
  }

  selectTool(toolKey) {
    this.selectedToolKey = toolKey;
  }

  findRun(runId) {
    if (runId && this.runs.has(runId)) return this.runs.get(runId);
    if (this.activeRunId && this.runs.has(this.activeRunId)) return this.runs.get(this.activeRunId);
    return null;
  }

  buildToolKey(runId, toolName, keyPart) {
    return `${runId || ''}::${toolName || 'tool'}::${keyPart || Date.now()}`;
  }

  toolLookupKey(runId, toolName) {
    return `${runId || ''}::${toolName || 'tool'}`;
  }

  callLookupKey(runId, toolCallId) {
    return `${runId || ''}::${toolCallId || ''}`;
  }

  pushActiveToolKey(runId, toolName, toolKey) {
    const lookupKey = this.toolLookupKey(runId, toolName);
    const activeKeys = this.activeToolKeysByLookup.get(lookupKey) || [];
    activeKeys.push(toolKey);
    this.activeToolKeysByLookup.set(lookupKey, activeKeys);
  }

  peekActiveToolKey(runId, toolName) {
    const activeKeys = this.activeToolKeysByLookup.get(this.toolLookupKey(runId, toolName)) || [];
    return activeKeys.length ? activeKeys[activeKeys.length - 1] : null;
  }

  removeActiveToolKey(runId, toolName, toolKey) {
    const lookupKey = this.toolLookupKey(runId, toolName);
    const activeKeys = this.activeToolKeysByLookup.get(lookupKey) || [];
    const nextKeys = activeKeys.filter((key) => key !== toolKey);
    if (nextKeys.length) {
      this.activeToolKeysByLookup.set(lookupKey, nextKeys);
    } else {
      this.activeToolKeysByLookup.delete(lookupKey);
    }
  }

  findToolByKey(toolKey) {
    if (!toolKey) return null;
    for (const runId of this.runOrder) {
      const run = this.runs.get(runId);
      if (!run) continue;
      for (const tool of run.tools) {
        if (tool.key === toolKey) return tool;
      }
    }
    return null;
  }

  findLatestTool(toolName) {
    for (const runId of this.runOrder) {
      const run = this.runs.get(runId);
      if (!run) continue;
      for (let i = run.tools.length - 1; i >= 0; i--) {
        if (run.tools[i].name === toolName) return run.tools[i];
      }
    }
    return null;
  }

  findLatestUnboundTool(toolName) {
    for (const runId of this.runOrder) {
      const run = this.runs.get(runId);
      if (!run) continue;
      for (let i = run.tools.length - 1; i >= 0; i--) {
        const tool = run.tools[i];
        if (tool.name === toolName && !tool.detailKey) return tool;
      }
    }
    return null;
  }

  takePendingDetailKey(toolName) {
    const queue = this.pendingDetailKeysByTool.get(toolName) || [];
    const detailKey = queue.shift() || null;
    if (queue.length) {
      this.pendingDetailKeysByTool.set(toolName, queue);
    } else {
      this.pendingDetailKeysByTool.delete(toolName);
    }
    return detailKey;
  }

  queuePendingDetailKey(toolName, detailKey) {
    const queue = this.pendingDetailKeysByTool.get(toolName) || [];
    queue.push(detailKey);
    this.pendingDetailKeysByTool.set(toolName, queue);
  }

  assignDetailKey(tool, detailKey) {
    if (!tool || !detailKey) return;
    if (tool.detailKey === tool.key || detailKey === tool.key) {
      tool.detailKey = tool.key;
      this.openDetailKeysByTool.set(tool.name, tool.key);
      return;
    }

    const existingMessages = this.detailsByToolCall.get(detailKey) || [];
    const targetMessages = this.detailsByToolCall.get(tool.key) || [];
    if (existingMessages.length) {
      this.detailsByToolCall.set(tool.key, targetMessages.concat(existingMessages));
      this.detailsByToolCall.delete(detailKey);
    }
    tool.detailKey = tool.key;
    if (this.openDetailKeysByTool.get(tool.name) === detailKey) {
      this.openDetailKeysByTool.set(tool.name, tool.key);
    }
    if (this.selectedToolKey === detailKey) {
      this.selectedToolKey = tool.key;
    }
  }

  resolveDetailKey(message) {
    const toolName = message.tool_name || 'tool';
    if (message.message_type === 'status') {
      const tool = this.findLatestUnboundTool(toolName);
      if (tool) {
        tool.detailKey = tool.key;
        this.openDetailKeysByTool.set(toolName, tool.key);
        return tool.key;
      }

      const pendingKey = `pending::${toolName}::${message.id || Date.now()}`;
      this.queuePendingDetailKey(toolName, pendingKey);
      this.openDetailKeysByTool.set(toolName, pendingKey);
      return pendingKey;
    }

    const openKey = this.openDetailKeysByTool.get(toolName);
    if (openKey) return openKey;

    const latestTool = this.findLatestTool(toolName);
    if (latestTool?.detailKey) return latestTool.detailKey;
    if (latestTool) {
      latestTool.detailKey = latestTool.key;
      this.openDetailKeysByTool.set(toolName, latestTool.key);
      return latestTool.key;
    }

    const fallbackKey = `pending::${toolName}::${message.id || Date.now()}`;
    this.openDetailKeysByTool.set(toolName, fallbackKey);
    return fallbackKey;
  }
}

class TimelineView {
  constructor(container, onSelectTool) {
    this.container = container;
    this.onSelectTool = onSelectTool;
  }

  render(state) {
    this.container.replaceChildren();
    if (!state.runOrder.length) {
      this.container.appendChild(emptyState('No activity'));
      return;
    }

    for (const runId of state.runOrder) {
      const run = state.runs.get(runId);
      if (!run) continue;
      this.container.appendChild(this.renderRun(run, state.selectedToolKey));
    }
  }

  renderRun(run, selectedToolKey) {
    const item = document.createElement('article');
    item.className = `timeline-run timeline-run--${run.status}`;

    const header = document.createElement('div');
    header.className = 'timeline-run__header';
    header.appendChild(textEl('span', 'timeline-run__time', formatTime(run.timestamp)));
    header.appendChild(textEl('span', `badge badge--${run.status}`, run.status));
    item.appendChild(header);

    item.appendChild(textEl('div', 'timeline-run__prompt', run.message || 'Agent run'));

    if (run.tools.length) {
      const tools = document.createElement('div');
      tools.className = 'timeline-tools';
      for (const tool of run.tools) {
        tools.appendChild(this.renderTool(tool, selectedToolKey));
      }
      item.appendChild(tools);
    }

    return item;
  }

  renderTool(tool, selectedToolKey) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = `timeline-tool timeline-tool--${tool.status}`;
    if (tool.key === selectedToolKey) row.classList.add('timeline-tool--selected');
    row.addEventListener('click', () => this.onSelectTool(tool.key));

    row.appendChild(textEl('span', 'timeline-tool__name', tool.name));
    const detail = tool.durationMs ? `${tool.durationMs}ms` : tool.status;
    row.appendChild(textEl('span', `timeline-tool__status badge badge--${tool.status}`, detail));
    return row;
  }
}

class DetailView {
  constructor(container) {
    this.container = container;
  }

  render(state) {
    this.container.replaceChildren();
    const tool = state.findToolByKey(state.selectedToolKey);
    if (!tool) {
      this.container.appendChild(emptyState('No tool selected'));
      return;
    }

    const detailKey = tool.detailKey || tool.key;
    const messages = state.detailsByToolCall.get(detailKey) || [];
    const title = textEl('div', 'detail-title', tool.name);
    this.container.appendChild(title);

    if (!messages.length) {
      if (tool?.images?.length || tool?.args != null || tool?.resultPreview || tool?.error) {
        this.renderToolFallback(tool);
      } else {
        this.container.appendChild(emptyState('No detail stream'));
      }
      return;
    }

    this.renderPlotly(messages);
    this.renderMedia(messages);
    this.renderSteps(messages);
    this.renderMessages(messages);
  }

  renderToolFallback(tool) {
    if (tool.images?.length) {
      const section = sectionEl('Vision');
      for (const image of tool.images) {
        const canvas = document.createElement('canvas');
        canvas.className = 'detail-canvas';
        canvas.width = 640;
        canvas.height = 480;
        section.appendChild(canvas);
        drawVisionCanvas(canvas, image, null);
      }
      this.container.appendChild(section);
    }
    if (tool.args != null) {
      const section = sectionEl('Arguments');
      section.appendChild(textEl('pre', 'detail-message__payload', formatPayload(tool.args)));
      this.container.appendChild(section);
    }
    if (tool.resultPreview || tool.error) {
      const section = sectionEl(tool.error ? 'Error' : 'Result');
      section.appendChild(textEl('pre', 'detail-message__payload', tool.error || tool.resultPreview));
      this.container.appendChild(section);
    }
  }

  renderPlotly(messages) {
    const msg = findLast(messages, (m) => m.message_type === 'plotly');
    if (!msg) return;

    const section = sectionEl('Embedding Space');
    const div = document.createElement('div');
    div.className = 'detail-plotly';
    section.appendChild(div);
    this.container.appendChild(section);

    const figure = msg.payload?.figure || {};
    Plotly.react(div, figure.data || [], figure.layout || {}, { responsive: true });
  }

  renderMedia(messages) {
    const imageMessage = findLast(messages, (message) => message.message_type === 'image');
    const detectionsMessage = findLast(messages, (message) => message.message_type === 'detections');
    if (!imageMessage && !detectionsMessage) return;

    const section = sectionEl('Vision');
    const canvas = document.createElement('canvas');
    canvas.className = 'detail-canvas';
    canvas.width = 640;
    canvas.height = 480;
    section.appendChild(canvas);
    this.container.appendChild(section);

    drawVisionCanvas(canvas, imageMessage?.payload, detectionsMessage?.payload);

    const points2d = detectionsMessage?.payload?.points_2d || [];
    if (points2d.length) section.appendChild(points2dTable(points2d));

    const points3d = detectionsMessage?.payload?.points_3d || [];
    if (points3d.length) section.appendChild(pointsTable(points3d));
  }

  renderSteps(messages) {
    const steps = messages.filter((message) => message.message_type === 'step');
    if (!steps.length) return;

    const section = sectionEl('Steps');
    const list = document.createElement('div');
    list.className = 'step-list';
    for (const message of steps.slice(-8)) {
      const row = document.createElement('div');
      row.className = 'step-row';
      row.appendChild(textEl('span', 'step-row__name', message.payload?.name || message.title || 'step'));
      row.appendChild(textEl('span', 'step-row__time', `${message.payload?.duration_ms ?? ''}ms`));
      list.appendChild(row);
    }
    section.appendChild(list);
    this.container.appendChild(section);
  }

  renderMessages(messages) {
    const section = sectionEl('Messages');
    for (const message of messages.slice(-12).reverse()) {
      if (message.message_type === 'image' || message.message_type === 'detections'
          || message.message_type === 'step' || message.message_type === 'plotly') {
        continue;
      }
      const item = document.createElement('div');
      item.className = `detail-message detail-message--${message.message_type}`;
      item.appendChild(textEl('div', 'detail-message__meta', `${formatTime(message.timestamp)} ${message.message_type}`));
      item.appendChild(textEl('pre', 'detail-message__payload', formatPayload(message.payload)));
      section.appendChild(item);
    }
    this.container.appendChild(section);
  }
}

class App {
  constructor() {
    this.state = new AppState();
    this.timelineView = new TimelineView(document.getElementById('timeline'), (toolKey) => this.selectTool(toolKey));
    this.detailView = new DetailView(document.getElementById('tool-details'));
    this.timelineStatus = document.getElementById('timeline-status');
    this.detailStatus = document.getElementById('detail-status');
    this.timelineSocket = new TimelineSocket(this);
    this.detailSocket = new ToolDetailSocket(this);
  }

  start() {
    this.render();
    this.timelineSocket.connect();
    this.detailSocket.connect();
  }

  handleTimelineEvent(event) {
    if (!event.id || this.state.seenTimelineIds.has(event.id)) return;
    this.state.seenTimelineIds.add(event.id);
    console.log(`[timeline] ${event.event_type}`, event.payload);

    const payload = event.payload || {};
    if (event.event_type === 'run.started') {
      this.state.upsertRun(event.run_id, payload.message || '', event.timestamp);
    } else if (event.event_type === 'run.completed') {
      this.state.finishRun(event.run_id, 'done', payload.content || '');
    } else if (event.event_type === 'run.error') {
      this.state.finishRun(event.run_id, 'error', payload.error || '');
    } else if (event.event_type === 'run.content') {
      this.state.addToken(event.run_id, payload.token || '');
    } else if (event.event_type === 'tool.call_started') {
      this.state.startTool(event.run_id, payload.tool, payload.args, event.id, payload.tool_call_id);
    } else if (event.event_type === 'tool.call_completed') {
      this.state.finishTool(event.run_id, payload.tool, 'done', payload);
    } else if (event.event_type === 'tool.call_error') {
      this.state.finishTool(event.run_id, payload.tool, 'error', payload);
    }

    this.render();
  }

  handleToolDetail(message) {
    if (!message.id || this.state.seenDetailIds.has(message.id)) return;
    this.state.seenDetailIds.add(message.id);
    console.log(`[detail] ${message.tool_name}/${message.message_type}`, message.payload);
    const detailKey = this.state.addDetail(message);
    const tool = this.state.findToolByDetailKey(detailKey) || this.state.findLatestTool(message.tool_name);
    if (tool?.key) this.state.selectTool(tool.key);
    this.render();
  }

  selectTool(toolKey) {
    this.state.selectTool(toolKey);
    this.render();
  }

  setTimelineStatus(connected) {
    console.log(`[gui] timeline ${connected ? 'connected' : 'disconnected'}`);
    setStatus(this.timelineStatus, connected);
  }

  setDetailStatus(connected) {
    console.log(`[gui] detail ${connected ? 'connected' : 'disconnected'}`);
    setStatus(this.detailStatus, connected);
  }

  render() {
    this.timelineView.render(this.state);
    this.detailView.render(this.state);
  }
}

function drawVisionCanvas(canvas, imagePayload, detectionsPayload) {
  const context = canvas.getContext('2d');
  const drawOverlay = () => drawDetections(context, canvas.width, canvas.height, detectionsPayload?.points_2d || []);

  context.fillStyle = '#f1f3f4';
  context.fillRect(0, 0, canvas.width, canvas.height);

  if (!imagePayload?.data) {
    drawOverlay();
    return;
  }

  const image = new Image();
  image.onload = () => {
    canvas.width = image.naturalWidth || 640;
    canvas.height = image.naturalHeight || 480;
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    drawOverlay();
  };
  image.src = `data:${imagePayload.mime_type || 'image/png'};base64,${imagePayload.data}`;
}

function drawDetections(context, width, height, points) {
  const colors = ['#1a73e8', '#188038', '#f29900', '#d93025', '#9334e6'];
  points.forEach((point, index) => {
    const x = ((point.x || 0) / 1000) * width;
    const y = ((point.y || 0) / 1000) * height;
    const color = colors[index % colors.length];
    context.beginPath();
    context.arc(x, y, 8, 0, 2 * Math.PI);
    context.fillStyle = color;
    context.globalAlpha = 0.88;
    context.fill();
    context.globalAlpha = 1;
    context.strokeStyle = '#ffffff';
    context.lineWidth = 2;
    context.stroke();

    const label = point.label || String(index + 1);
    context.font = '600 13px Arial, sans-serif';
    const textWidth = context.measureText(label).width + 10;
    context.fillStyle = color;
    context.fillRect(x + 10, y - 11, textWidth, 20);
    context.fillStyle = '#ffffff';
    context.fillText(label, x + 15, y + 4);
  });
}

function pointsTable(points) {
  const table = document.createElement('div');
  table.className = 'points-table';
  points.forEach((point, index) => {
    const row = document.createElement('div');
    row.className = 'points-row';
    row.appendChild(textEl('span', 'points-row__index', String(index + 1)));
    row.appendChild(textEl('span', 'points-row__name', point.label || 'point'));
    row.appendChild(textEl('span', 'points-row__coords', `(${fmt(point.x)}, ${fmt(point.y)}, ${fmt(point.z)}) m`));
    table.appendChild(row);
  });
  return table;
}

function points2dTable(points) {
  const table = document.createElement('div');
  table.className = 'points-table';
  points.forEach((point, index) => {
    const row = document.createElement('div');
    row.className = 'points-row';
    row.appendChild(textEl('span', 'points-row__index', String(index + 1)));
    row.appendChild(textEl('span', 'points-row__name', point.label || 'point'));
    row.appendChild(textEl('span', 'points-row__coords', `(${fmt(point.x)}, ${fmt(point.y)})`));
    table.appendChild(row);
  });
  return table;
}

function sectionEl(title) {
  const section = document.createElement('section');
  section.className = 'detail-section';
  section.appendChild(textEl('h3', 'detail-section__title', title));
  return section;
}

function emptyState(text) {
  return textEl('div', 'empty-state', text);
}

function textEl(tagName, className, text) {
  const element = document.createElement(tagName);
  element.className = className;
  element.textContent = String(text ?? '');
  return element;
}

function setStatus(element, connected) {
  element.className = `status-pill status-pill--${connected ? 'connected' : 'disconnected'}`;
  element.textContent = connected ? 'connected' : 'reconnecting';
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function findLast(items, predicate) {
  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (predicate(items[index])) return items[index];
  }
  return null;
}

function formatTime(timestamp) {
  if (!timestamp) return '';
  return new Date(timestamp * 1000).toLocaleTimeString();
}

function formatPayload(payload) {
  try {
    return JSON.stringify(payload || {}, null, 2);
  } catch {
    return String(payload || '');
  }
}

function fmt(value) {
  return typeof value === 'number' ? value.toFixed(3) : String(value ?? '');
}

new App().start();
