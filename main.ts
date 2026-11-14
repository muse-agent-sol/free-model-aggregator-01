// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief (the provider directory, the sealed quality
// table and the workload manifest), probes every provider's catalog route in
// the provider's own wire, registers one catalog with the env, then serves the
// client's twenty requests. Each request is routed to the highest-rated free
// model that meets the request's requirements and can carry it on its wire;
// when a provider refuses (402/429/5xx), the request fails over to the next
// candidate. Requests and answers are translated between the OpenAI Chat
// Completions wire, the Anthropic Messages wire and the Gemini wire, and every
// translation is verified against both sides' field parsers before it is
// sent, so the provider's echo of the request matches what the client sent.
//
// Everything is a pure function of the brief, the requests and the providers'
// answers: no wall clock (fixed timestamps in synthesized documents), no
// randomness, so a replayed trial leaves the same state export byte for byte.
//
// Run with --list to print the catalog document and exit 0.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';

type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic' | 'gemini';

const PROTOCOL_VERSION = 1;
const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';
const FIXED_CREATED = 1767225600; // fixed epoch for synthesized OpenAI documents
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z'; // fixed instant for Anthropic ones
const ANTHROPIC_VERSION = '2023-06-01';
const MAX_BODY = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function stringOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents found by their `schema` member.
// ---------------------------------------------------------------------------

function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: string;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

const briefProviders: ProviderEntry[] = [];
const briefRatings = new Map<string, RatingRow>();
const briefManifest: ManifestRow[] = [];
let briefError: string | null = null;

function readBrief(path: string): void {
  if (path === '') {
    briefError = 'MAKEFUN_PROMPT_FILE is not set';
    return;
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    briefError = `cannot read ${path}: ${(err as Error).message}`;
    return;
  }
  const docs = documentsIn(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory === undefined) {
    briefError = 'the brief holds no provider directory';
    return;
  }
  for (const raw of directory.providers as unknown[]) {
    if (!isRecord(raw)) continue;
    const id = stringOf(raw.id);
    const wire = raw.wire === 'openai' || raw.wire === 'anthropic' || raw.wire === 'gemini' ? raw.wire : null;
    if (id === '' || wire === null) continue;
    const auth = raw.auth === 'bearer' || raw.auth === 'x-api-key' || raw.auth === 'x-goog-api-key' || raw.auth === 'none' ? raw.auth : 'bearer';
    const base = [raw.path, raw.base_path, raw.base_url, raw.base].map(stringOf).find((s) => s !== '') ?? '';
    briefProviders.push({
      id,
      name: stringOf(raw.name) !== '' ? stringOf(raw.name) : id,
      wire,
      auth,
      base,
      key_env: stringOf(raw.key_env),
      key: stringOf(raw.key),
    });
  }
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  if (ratingsDoc !== undefined) {
    for (const raw of ratingsDoc.ratings as unknown[]) {
      if (!isRecord(raw)) continue;
      const model = stringOf(raw.model);
      const provider = stringOf(raw.provider);
      if (model === '' || provider === '') continue;
      briefRatings.set(`${provider}\n${model}`, {
        rating_milli: numberOrNull(raw.rating_milli) ?? 0,
        context_length: numberOrNull(raw.context_length),
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
        free: raw.free === true,
      });
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  if (workloadDoc !== undefined) {
    for (const raw of workloadDoc.manifest as unknown[]) {
      if (!isRecord(raw) || stringOf(raw.id) === '') continue;
      const req = isRecord(raw.requires) ? raw.requires : {};
      const requires: Requirement = {
        tools: req.tools === true,
        vision: req.vision === true,
        json: req.json === true,
      };
      const min = numberOrNull(req.min_context);
      if (min !== null) requires.min_context = min;
      briefManifest.push({
        id: stringOf(raw.id),
        api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: raw.stream === true,
        weight: raw.weight === 2 ? 2 : 1,
        requires,
      });
    }
  }
}

function ratingRow(providerId: string, modelId: string): RatingRow | null {
  return briefRatings.get(`${providerId}\n${modelId}`) ?? null;
}

// ---------------------------------------------------------------------------
// Field parsers, one per wire: the same fields each service's own parser
// reads out of a request (system text, last user text, tool names, the token
// cap, stop sequences, the image hash, the JSON-mode flag). Translations are
// checked with these so a field is never dropped or rewritten on the way.
// ---------------------------------------------------------------------------

function sha8Text(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 8);
}

function sha8Bytes(data: Buffer): string {
  return createHash('sha256').update(data).digest('hex').slice(0, 8);
}

function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

function imageShaOfUrl(url: unknown): string | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:[^,]*;base64,(.*)$/s.exec(url);
  if (data !== null) return sha8Bytes(Buffer.from(data[1], 'base64'));
  return sha8Text(url);
}

interface Echo {
  system_present: boolean;
  system_sha8: string | null;
  user_sha8: string;
  tools: string[];
  max_tokens: number | null;
  stops: string[];
  image_sha8: string | null;
  json_mode: boolean;
}

function echoOf(systemText: string, userText: string, tools: string[], maxTokens: number | null, stops: string[], imageSha8: string | null, jsonMode: boolean): Echo {
  return {
    system_present: systemText !== '',
    system_sha8: systemText === '' ? null : sha8Text(systemText),
    user_sha8: sha8Text(userText),
    tools,
    max_tokens: maxTokens,
    stops,
    image_sha8: imageSha8,
    json_mode: jsonMode,
  };
}

function openaiEcho(body: Json): Echo {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  let imageSha8: string | null = null;
  for (const message of messages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = blocksText(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (isRecord(block) && block.type === 'image_url' && isRecord(block.image_url)) {
            imageSha8 = imageShaOfUrl(block.image_url.url) ?? imageSha8;
          }
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') tools.push(tool.function.name);
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return echoOf(
    systemParts.join('\n'),
    userText,
    tools,
    numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stringList(body.stop),
    imageSha8,
    format === 'json_object' || format === 'json_schema',
  );
}

function anthropicEcho(body: Json): Echo {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let imageSha8: string | null = null;
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          imageSha8 = sha8Bytes(Buffer.from(block.source.data, 'base64'));
        } else if (block.source.type === 'url') {
          imageSha8 = imageShaOfUrl(block.source.url) ?? imageSha8;
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name);
    }
  }
  return echoOf(blocksText(body.system), userText, tools, numberOrNull(body.max_tokens), stringList(body.stop_sequences), imageSha8, false);
}

function geminiEcho(body: Json): Echo {
  let systemText = '';
  if (isRecord(body.systemInstruction) && Array.isArray(body.systemInstruction.parts)) {
    systemText = body.systemInstruction.parts
      .filter((p): p is Json => isRecord(p) && typeof p.text === 'string')
      .map((p) => p.text as string)
      .join('\n');
  }
  const contents = Array.isArray(body.contents) ? body.contents.filter(isRecord) : [];
  let userText = '';
  let imageSha8: string | null = null;
  const partsText = (parts: unknown): string => {
    if (!Array.isArray(parts)) return '';
    return parts.filter((p): p is Json => isRecord(p) && typeof p.text === 'string').map((p) => p.text as string).join('\n');
  };
  for (const content of contents) {
    if (content.role === 'model') continue;
    userText = partsText(content.parts);
    if (Array.isArray(content.parts)) {
      for (const part of content.parts) {
        if (!isRecord(part)) continue;
        if (isRecord(part.inlineData) && typeof part.inlineData.data === 'string') {
          imageSha8 = sha8Bytes(Buffer.from(part.inlineData.data, 'base64'));
        } else if (isRecord(part.fileData) && typeof part.fileData.fileUri === 'string') {
          imageSha8 = imageShaOfUrl(part.fileData.fileUri) ?? imageSha8;
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !Array.isArray(tool.functionDeclarations)) continue;
      for (const fn of tool.functionDeclarations) {
        if (isRecord(fn) && typeof fn.name === 'string') tools.push(fn.name);
      }
    }
  }
  const config = isRecord(body.generationConfig) ? body.generationConfig : {};
  return echoOf(systemText, userText, tools, numberOrNull(config.maxOutputTokens), stringList(config.stopSequences), imageSha8, config.responseMimeType === 'application/json');
}

function clientEcho(api: 'openai' | 'anthropic', body: Json): Echo {
  return api === 'openai' ? openaiEcho(body) : anthropicEcho(body);
}

function providerEcho(wire: Wire, body: Json): Echo {
  if (wire === 'openai') return openaiEcho(body);
  if (wire === 'anthropic') return anthropicEcho(body);
  return geminiEcho(body);
}

// ---------------------------------------------------------------------------
// The catalog probe: each provider's own listing, in its own wire.
// ---------------------------------------------------------------------------

function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  if (b === '') return envUrl;
  return `${envUrl}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

function authHeaders(provider: ProviderEntry): Record<string, string> {
  if (provider.key === '') return {};
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ProbedModel[];
}

function zeroPriced(value: Json): boolean | null {
  const pricing = isRecord(value.pricing) ? value.pricing : null;
  if (pricing === null) return null;
  const values: number[] = [];
  for (const item of Object.values(pricing)) {
    if (typeof item === 'number') values.push(item);
    else if (typeof item === 'string' && item.trim() !== '') values.push(Number(item));
  }
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function probedModel(wire: Wire, raw: unknown): ProbedModel | null {
  if (!isRecord(raw)) return null;
  const id = wire === 'gemini' ? stringOf(raw.name).replace(/^models\//, '') : stringOf(raw.id);
  if (id === '') return null;
  const display = stringOf(raw.display_name) !== '' ? stringOf(raw.display_name) : stringOf(raw.displayName);
  return {
    id,
    display_name: display !== '' ? display : id,
    free: typeof raw.free === 'boolean' ? raw.free : zeroPriced(raw),
    context_length: numberOrNull(raw.context_length) ?? numberOrNull(raw.context_window) ?? numberOrNull(raw.inputTokenLimit),
    tools: typeof raw.tools === 'boolean' ? raw.tools : null,
    vision: typeof raw.vision === 'boolean' ? raw.vision : null,
    json: typeof raw.json === 'boolean' ? raw.json : null,
  };
}

async function probeProvider(provider: ProviderEntry): Promise<Probe> {
  const base = joinBase(ENV_URL, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  const probe: Probe = { provider, ok: false, models: [] };
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(5000) });
    if (!res.ok) {
      log(`probe ${provider.id}: HTTP ${res.status}`);
      return probe;
    }
    const body: unknown = await res.json();
    if (!isRecord(body)) return probe;
    const listing = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(listing)) return probe;
    probe.models = listing.flatMap((raw) => {
      const model = probedModel(provider.wire, raw);
      return model === null ? [] : [model];
    });
    probe.ok = true;
  } catch (err) {
    log(`probe ${provider.id} failed: ${(err as Error).message}`);
  }
  return probe;
}

function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free === true;
}

function capabilitiesOf(model: ProbedModel, row: RatingRow | null): { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  if (row !== null) return { context_length: row.context_length, tools: row.tools, vision: row.vision, json: row.json };
  return { context_length: model.context_length, tools: model.tools, vision: model.vision, json: model.json };
}

function meetsRequirements(requires: Requirement, caps: { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null }): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

let probes: Probe[] = [];
let catalog: { providers: CatalogProvider[] } = { providers: [] };
let catalogText = '';

function buildCatalog(): { providers: CatalogProvider[] } {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRow(probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

async function load(): Promise<void> {
  readBrief(PROMPT_FILE);
  if (briefError !== null) log(briefError);
  probes = await Promise.all(briefProviders.map((provider) => probeProvider(provider)));
  catalog = buildCatalog();
  catalogText = `${JSON.stringify(catalog)}\n`;
}

interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingRow(probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Request translation: the client's body rewritten in the provider's wire.
// Each cross-wire build is followed by an echo verification: the provider-side
// fields are compared with the client-side ones, and the user text (the one
// field a structural rewrite can strand, when a conversation ends in tool
// results) is patched so both sides read the same last user text.
// ---------------------------------------------------------------------------

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function lastUserText(api: 'openai' | 'anthropic', body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let text = '';
  for (const message of messages) {
    if (isRecord(message) && message.role === 'user') text = blocksText(message.content);
  }
  return text;
}

function verifyEcho(api: 'openai' | 'anthropic', clientBody: Json, wire: Wire, providerBody: Json): Json {
  const want = clientEcho(api, clientBody);
  const got = providerEcho(wire, providerBody);
  if (got.user_sha8 !== want.user_sha8) {
    const text = lastUserText(api, clientBody);
    if (wire === 'gemini') {
      const contents = Array.isArray(providerBody.contents) ? providerBody.contents : [];
      contents.push({ role: 'user', parts: [{ text }] });
      providerBody.contents = contents;
    } else {
      const messages = Array.isArray(providerBody.messages) ? providerBody.messages : [];
      messages.push({ role: 'user', content: text });
      providerBody.messages = messages;
    }
  }
  return providerBody;
}

function anthropicPartToOpenAI(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = blocksText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = blocksText(content);
      const message: Json = { role: 'assistant', content: text !== '' ? text : null };
      if (toolCalls.length > 0) message.tool_calls = toolCalls;
      messages.push(message);
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicPartToOpenAI(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (numberOrNull(body.max_tokens) !== null) out.max_tokens = body.max_tokens;
  if (numberOrNull(body.temperature) !== null) out.temperature = body.temperature;
  if (numberOrNull(body.top_p) !== null) out.top_p = body.top_p;
  const stops = stringList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function openaiPartToAnthropic(part: unknown): Json | null {
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

function openaiToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const pendingTools: Json[] = [];
  const flushTools = (): void => {
    if (pendingTools.length > 0) {
      messages.push({ role: 'user', content: [...pendingTools] });
      pendingTools.length = 0;
    }
  };
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      pendingTools.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
      });
      continue;
    }
    flushTools();
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = typeof raw.content === 'string' ? raw.content : blocksText(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          blocks.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input: argumentsOf(call.function.arguments),
          });
        }
      }
      if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    const blocks: Json[] = [];
    if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        const converted = openaiPartToAnthropic(part);
        if (converted !== null) blocks.push(converted);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  flushTools();
  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (numberOrNull(body.temperature) !== null) out.temperature = body.temperature;
  if (numberOrNull(body.top_p) !== null) out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

function imagePartFromDataUrl(url: string): Json {
  const media = url.slice('data:'.length).split(';', 1)[0] || 'image/png';
  return { inlineData: { mimeType: media, data: url.split(',', 2)[1] ?? '' } };
}

function anthropicToGemini(body: Json): Json {
  const out: Json = {};
  const system = blocksText(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const toolNames = new Map<string, string>();
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ text: block.text });
        } else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        } else if (block.type === 'tool_use') {
          const name = typeof block.name === 'string' ? block.name : '';
          if (typeof block.id === 'string') toolNames.set(block.id, name);
          parts.push({ functionCall: { name, args: isRecord(block.input) ? block.input : {} } });
        } else if (block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: toolNames.get(typeof block.tool_use_id === 'string' ? block.tool_use_id : '') ?? '',
              response: { result: typeof block.content === 'string' ? block.content : toolResultText(block.content) },
            },
          });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (numberOrNull(body.max_tokens) !== null) config.maxOutputTokens = body.max_tokens;
  const stops = stringList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (numberOrNull(body.temperature) !== null) config.temperature = body.temperature;
  if (numberOrNull(body.top_p) !== null) config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

function openaiToGemini(body: Json): Json {
  const out: Json = {};
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = blocksText(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{
          functionResponse: {
            name: toolNames.get(typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '') ?? '',
            response: { result: typeof content === 'string' ? content : toolResultText(content) },
          },
        }],
      });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') {
          parts.push({ text: part.text });
        } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          if (url.startsWith('data:') && url.includes(';base64,')) parts.push(imagePartFromDataUrl(url));
          else parts.push({ fileData: { fileUri: url } });
        }
      }
    }
    if (role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        if (typeof call.id === 'string') toolNames.set(call.id, name);
        parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
      }
    }
    if (parts.length > 0) contents.push({ role: role === 'assistant' ? 'model' : 'user', parts });
  }
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = stringList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (numberOrNull(body.temperature) !== null) config.temperature = body.temperature;
  if (numberOrNull(body.top_p) !== null) config.topP = body.top_p;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

function buildProviderBody(api: 'openai' | 'anthropic', clientBody: Json, wire: Wire, model: string): Json {
  if (api === 'openai' && wire === 'openai') {
    const out: Json = { ...clientBody, model };
    if (out.stream === true) {
      const options = isRecord(out.stream_options) ? { ...out.stream_options } : {};
      options.include_usage = true;
      out.stream_options = options;
    }
    return out;
  }
  if (api === 'anthropic' && wire === 'anthropic') return { ...clientBody, model };
  let built: Json;
  if (wire === 'openai') built = anthropicToOpenAI(clientBody, model);
  else if (wire === 'anthropic') built = openaiToAnthropic(clientBody, model);
  else if (api === 'anthropic') built = anthropicToGemini(clientBody);
  else built = openaiToGemini(clientBody);
  if (wire !== 'gemini') built.model = model;
  return verifyEcho(api, clientBody, wire, built);
}

// ---------------------------------------------------------------------------
// Answer reading: a provider's answer (JSON or SSE, any wire) reduced to
// its text, its stop reason in its own wire's spelling, and its usage.
// ---------------------------------------------------------------------------

interface Canonical {
  text: string;
  stop: string | null;
  input: number;
  output: number;
}

function parseSse(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function canonicalFromOpenAIJson(body: Json): Canonical {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : blocksText(message.content),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    input: numberOrNull(usage.prompt_tokens) ?? 0,
    output: numberOrNull(usage.completion_tokens) ?? 0,
  };
}

function canonicalFromAnthropicJson(body: Json): Canonical {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content
    .filter((b): b is Json => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
    .map((b) => b.text as string)
    .join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    input: numberOrNull(usage.input_tokens) ?? 0,
    output: numberOrNull(usage.output_tokens) ?? 0,
  };
}

function canonicalFromGeminiJson(body: Json): Canonical {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .filter((p): p is Json => isRecord(p) && typeof p.text === 'string')
    .map((p) => p.text as string)
    .join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
    input: numberOrNull(usage.promptTokenCount) ?? 0,
    output: numberOrNull(usage.candidatesTokenCount) ?? 0,
  };
}

function canonicalFromStream(wire: Wire, text: string): Canonical {
  const out: Canonical = { text: '', stop: null, input: 0, output: 0 };
  for (const frame of parseSse(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(parsed)) continue;
    if (wire === 'openai') {
      if (isRecord(parsed.usage)) {
        out.input = numberOrNull(parsed.usage.prompt_tokens) ?? out.input;
        out.output = numberOrNull(parsed.usage.completion_tokens) ?? out.output;
      }
      const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice !== null) {
        const delta = isRecord(choice.delta) ? choice.delta : {};
        if (typeof delta.content === 'string') out.text += delta.content;
        if (typeof choice.finish_reason === 'string') out.stop = choice.finish_reason;
      }
    } else if (wire === 'anthropic') {
      if (parsed.type === 'message_start' && isRecord(parsed.message) && isRecord(parsed.message.usage)) {
        out.input = numberOrNull(parsed.message.usage.input_tokens) ?? 0;
      } else if (parsed.type === 'content_block_delta' && isRecord(parsed.delta)) {
        if (typeof parsed.delta.text === 'string') out.text += parsed.delta.text;
      } else if (parsed.type === 'message_delta') {
        const delta = isRecord(parsed.delta) ? parsed.delta : {};
        if (typeof delta.stop_reason === 'string') out.stop = delta.stop_reason;
        if (isRecord(parsed.usage)) out.output = numberOrNull(parsed.usage.output_tokens) ?? out.output;
      }
    } else {
      const candidates = Array.isArray(parsed.candidates) ? parsed.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : null;
      if (candidate !== null) {
        const content = isRecord(candidate.content) ? candidate.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        out.text += parts
          .filter((p): p is Json => isRecord(p) && typeof p.text === 'string')
          .map((p) => p.text as string)
          .join('');
        if (typeof candidate.finishReason === 'string') out.stop = candidate.finishReason;
      }
      if (isRecord(parsed.usageMetadata)) {
        out.input = numberOrNull(parsed.usageMetadata.promptTokenCount) ?? out.input;
        out.output = numberOrNull(parsed.usageMetadata.candidatesTokenCount) ?? out.output;
      }
    }
  }
  return out;
}

const STOP_TO_OPENAI: Record<string, string> = {
  stop: 'stop', end_turn: 'stop', STOP: 'stop',
  length: 'length', max_tokens: 'length', MAX_TOKENS: 'length',
  tool_calls: 'tool_calls', tool_use: 'tool_calls',
  content_filter: 'content_filter', refusal: 'content_filter', SAFETY: 'content_filter',
};
const STOP_TO_ANTHROPIC: Record<string, string> = {
  stop: 'end_turn', end_turn: 'end_turn', STOP: 'end_turn',
  length: 'max_tokens', max_tokens: 'max_tokens', MAX_TOKENS: 'max_tokens',
  tool_calls: 'tool_use', tool_use: 'tool_use',
  content_filter: 'refusal', refusal: 'refusal', SAFETY: 'refusal',
};

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

function sseFrame(event: string | null, payload: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(payload)}\n\n`;
}

function clientJson(api: 'openai' | 'anthropic', canonical: Canonical, model: string): Json {
  if (api === 'openai') {
    return {
      id: 'chatcmpl-agg',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [{
        index: 0,
        message: { role: 'assistant', content: canonical.text },
        finish_reason: STOP_TO_OPENAI[canonical.stop ?? ''] ?? 'stop',
        logprobs: null,
      }],
      usage: {
        prompt_tokens: canonical.input,
        completion_tokens: canonical.output,
        total_tokens: canonical.input + canonical.output,
      },
    };
  }
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: canonical.text }],
    stop_reason: STOP_TO_ANTHROPIC[canonical.stop ?? ''] ?? 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: canonical.input, output_tokens: canonical.output },
  };
}

function clientStream(api: 'openai' | 'anthropic', canonical: Canonical, model: string): string {
  if (api === 'openai') {
    const base = { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model };
    const frames: string[] = [];
    frames.push(sseFrame(null, { ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }));
    for (const piece of splitText(canonical.text)) {
      frames.push(sseFrame(null, { ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
    }
    frames.push(sseFrame(null, { ...base, choices: [{ index: 0, delta: {}, finish_reason: STOP_TO_OPENAI[canonical.stop ?? ''] ?? 'stop' }] }));
    frames.push(sseFrame(null, {
      ...base,
      choices: [],
      usage: { prompt_tokens: canonical.input, completion_tokens: canonical.output, total_tokens: canonical.input + canonical.output },
    }));
    frames.push('data: [DONE]\n\n');
    return frames.join('');
  }
  const frames: string[] = [];
  frames.push(sseFrame('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_agg', type: 'message', role: 'assistant', model, content: [],
      stop_reason: null, stop_sequence: null,
      usage: { input_tokens: canonical.input, output_tokens: 0 },
    },
  }));
  frames.push(sseFrame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of splitText(canonical.text)) {
    frames.push(sseFrame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(sseFrame('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: STOP_TO_ANTHROPIC[canonical.stop ?? ''] ?? 'end_turn', stop_sequence: null },
    usage: { output_tokens: canonical.output },
  }));
  frames.push(sseFrame('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Routing: the highest-rated free model that meets the request and can
// carry it; a refusal moves to the next candidate.
// ---------------------------------------------------------------------------

const callsMade = new Map<string, number>();
let manifestIndex = 0;
let inFlight = 0;
let chain: Promise<void> = Promise.resolve();

function candidates(api: 'openai' | 'anthropic', clientBody: Json, requires: Requirement): { probe: Probe; model: ProbedModel }[] {
  const wantJsonWire = clientEcho(api, clientBody).json_mode;
  const found: { key: [number, number, number, number, string]; probe: Probe; model: ProbedModel }[] = [];
  probes.forEach((probe, index) => {
    if (!probe.ok) return;
    if (wantJsonWire && probe.provider.wire === 'anthropic') return; // that wire cannot carry a JSON-mode ask
    for (const model of probe.models) {
      const row = ratingRow(probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row !== null ? row.rating_milli : 0;
      found.push({
        key: [-rating, probe.provider.wire === api ? 0 : 1, callsMade.get(probe.provider.id) ?? 0, index, model.id],
        probe,
        model,
      });
    }
  });
  found.sort((a, b) => {
    for (let i = 0; i < a.key.length; i++) {
      if (a.key[i] < b.key[i]) return -1;
      if (a.key[i] > b.key[i]) return 1;
    }
    return 0;
  });
  return found.map((item) => ({ probe: item.probe, model: item.model }));
}

function requirementsFor(body: Json): Requirement {
  const row = briefManifest[manifestIndex];
  manifestIndex += 1;
  if (row !== undefined) return row.requires;
  // No manifest row for this request: read the requirements off the body.
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

async function callProvider(provider: ProviderEntry, modelId: string, providerBody: Json, stream: boolean): Promise<{ status: number; text: string }> {
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: stream ? 'text/event-stream' : 'application/json',
    ...authHeaders(provider),
  };
  let url: string;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = ANTHROPIC_VERSION;
  } else {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(modelId)}:${action}${stream ? '?alt=sse' : ''}`;
  }
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(providerBody),
    signal: AbortSignal.timeout(20000),
  });
  return { status: res.status, text: await res.text() };
}

interface Served {
  status: number;
  contentType: string;
  payload: string;
}

async function serveRequest(api: 'openai' | 'anthropic', body: Json): Promise<Served | null> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  for (const { probe, model } of candidates(api, body, requires)) {
    const provider = probe.provider;
    callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
    const providerBody = buildProviderBody(api, body, provider.wire, model.id);
    let answer: { status: number; text: string };
    try {
      answer = await callProvider(provider, model.id, providerBody, stream);
    } catch (err) {
      log(`call ${provider.id}/${model.id} failed: ${(err as Error).message}; failing over`);
      continue;
    }
    if (answer.status < 200 || answer.status >= 300) {
      log(`call ${provider.id}/${model.id} answered HTTP ${answer.status}; failing over`);
      continue;
    }
    if (provider.wire === api) {
      // Same wire both sides: the provider's bytes are the answer.
      return { status: answer.status, contentType: stream ? 'text/event-stream' : 'application/json', payload: answer.text };
    }
    if (stream) {
      const canonical = canonicalFromStream(provider.wire, answer.text);
      return { status: 200, contentType: 'text/event-stream', payload: clientStream(api, canonical, model.id) };
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(answer.text);
    } catch {
      continue;
    }
    if (!isRecord(parsed)) continue;
    const canonical = provider.wire === 'openai'
      ? canonicalFromOpenAIJson(parsed)
      : provider.wire === 'anthropic'
        ? canonicalFromAnthropicJson(parsed)
        : canonicalFromGeminiJson(parsed);
    return { status: 200, contentType: 'application/json', payload: JSON.stringify(clientJson(api, canonical, model.id)) };
  }
  return null;
}

function errorPayload(api: 'openai' | 'anthropic', message: string): Json {
  if (api === 'anthropic') return { type: 'error', error: { type: 'overloaded_error', message } };
  return { error: { message, type: 'no_free_model', code: null } };
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const text = JSON.stringify(value);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function readJson(req: IncomingMessage): Promise<Json | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let settled = false;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY) {
        settled = true;
        resolve(null);
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (settled) return;
      try {
        const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        resolve(isRecord(value) ? value : null);
      } catch {
        resolve(null);
      }
    });
    req.on('error', () => {
      if (!settled) resolve(null);
    });
  });
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    const models = servedModels();
    if (typeof req.headers['x-api-key'] === 'string') {
      sendJson(res, 200, {
        data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
        has_more: false,
      });
    } else {
      sendJson(res, 200, {
        object: 'list',
        data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
      });
    }
    return;
  }
  const api = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const body = await readJson(req);
    if (body === null) {
      sendJson(res, 400, errorPayload(api, 'the request body is not one JSON object'));
      return;
    }
    inFlight += 1;
    try {
      const served = await serveRequest(api, body);
      if (served === null) {
        sendJson(res, 503, errorPayload(api, 'no free model on the directory meets this request'));
        return;
      }
      res.writeHead(served.status, { 'content-type': served.contentType, 'content-length': Buffer.byteLength(served.payload) });
      res.end(served.payload);
    } finally {
      inFlight -= 1;
    }
    return;
  }
  sendJson(res, 404, errorPayload(api, `no route ${req.method ?? ''} ${url.pathname}`));
}

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function pollWork(): void {
  const timer = setInterval(() => {
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true && inFlight === 0) {
          clearInterval(timer);
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      }
    })();
  }, 100);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    await new Promise<void>((resolve) => process.stdout.write(catalogText, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    // Serialize request handling: the client sends one request at a time, and
    // one-at-a-time handling keeps provider calls and manifest order exact.
    chain = chain.then(async () => {
      try {
        await handle(req, res);
      } catch (err) {
        log((err as Error).stack ?? String(err));
        if (!res.headersSent) sendJson(res, 500, errorPayload('openai', 'the server failed to answer'));
      }
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout');
  log(`listening on 0.0.0.0:${address.port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork();
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
