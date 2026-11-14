// The upstream wires the starter does not speak: Anthropic and Gemini, in both
// directions, JSON and server-sent events.
//
// Everything here is a pure function of its arguments (or a generator over one
// provider stream): no clock, no randomness, so a live pass and its replay
// produce the same bytes. The one trick that matters for the wire checks:
//
//   - the provider's answer text (the `ANSWER ECHO...` echo of what it parsed)
//     is passed through VERBATIM as the response's text. The driver's checks
//     compare that echo with what the client sent, so no field may be rewritten
//     or dropped in translation: system prompt, user text, images, tools,
//     stop sequences, max_tokens and JSON mode all survive, in whichever wire
//     carries them.
//   - a streamed OpenAI answer needs its usage chunk for the driver's usage
//     check, and the client never asks for one (no stream_options in the
//     workload), so every upstream OpenAI stream call below sets
//     stream_options.include_usage, and every translated stream ends with a
//     usage chunk before [DONE].
//   - JSON mode can only travel on the OpenAI wire (response_format) and the
//     Gemini wire (responseMimeType). The Anthropic wire cannot carry it, so
//     the router never sends a JSON-mode request to an Anthropic-wire provider.

import { isRecord, type Json } from './providers.ts';
import { openAIToAnthropic, sseEvent, stopReasonOf } from './wire.ts';

/** A fixed epoch second for every completion this file builds: nothing reads the clock. */
export const FIXED_CREATED = 1767225600;

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringListOf(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.filter((v): v is string => typeof v === 'string');
  return [];
}

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/**
 * An OpenAI image_url part as an Anthropic image block (base64 data URL) or a
 * file URL block; null when the part is not an image.
 */
function anthropicImageOf(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  }
  return { type: 'image', source: { type: 'url', url } };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`. System messages become the system string, image_url parts become
 * image blocks, tool_calls become tool_use blocks, tool messages become
 * tool_result blocks, and max_tokens, temperature, top_p, stop sequences,
 * tools, tool_choice and streaming pass through. response_format is dropped:
 * the Anthropic wire cannot carry JSON mode.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (role === 'assistant') {
      const toolCalls: Json[] = [];
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let input: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) input = parsed;
            } catch {
              // Not JSON: an empty input.
            }
          }
          toolCalls.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input,
          });
        }
      }
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      blocks.push(...toolCalls);
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      const text = typeof content === 'string' ? content : textOf(content);
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: text }],
      });
      continue;
    }
    // user (and anything else): text and image parts become Anthropic blocks.
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      const image = anthropicImageOf(part);
      if (image !== null) blocks.push(image);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = numberOrZero(body.max_tokens) || numberOrZero(body.max_completion_tokens);
  if (maxTokens > 0) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringListOf(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** The text of an Anthropic message's content blocks, joined like the driver's parser does. */
function anthropicTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** A Chat Completions answer object from an Anthropic message's JSON body. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content: unknown = body.content;
  const text = anthropicTextOf(content);
  const toolCalls: Json[] = [];
  if (Array.isArray(content)) {
    for (const block of content) {
      if (!isRecord(block) || block.type !== 'tool_use') continue;
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: anthropicStopToOpenAI(body.stop_reason),
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** One OpenAI SSE chunk line: a content delta, a finish chunk or the usage chunk. */
function openAIChunk(model: string, delta: Json, finish: string | null): string {
  return `data: ${JSON.stringify({ id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
}

function openAIUsageChunk(model: string, input: number, output: number): string {
  return `data: ${JSON.stringify({ id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output } })}\n\n`;
}

/**
 * An Anthropic SSE stream as OpenAI chat-completion chunks: text deltas become
 * content deltas, input_json_delta fragments become tool-call argument deltas,
 * message_delta's stop reason becomes the finish chunk, and the stream ends
 * with a usage chunk and [DONE]. The input tokens come from message_start,
 * the output tokens from message_delta.
 */
export async function* anthropicEventsToOpenAIChunks(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let finish: string | null = null;
  let sawFinish = false;
  const toolIndexes = new Map<number, string>();

  const frameEvents = function* (frame: string): Generator<string> {
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
    }
    if (dataLines.length === 0) return;
    let data: unknown;
    try {
      data = JSON.parse(dataLines.join('\n'));
    } catch {
      return;
    }
    if (!isRecord(data)) return;
    if (event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
      return;
    }
    if (event === 'content_block_delta' && isRecord(data.delta)) {
      const index = typeof data.index === 'number' ? data.index : 0;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        yield openAIChunk(model, { content: data.delta.text }, null);
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
        const id = toolIndexes.get(index) ?? `toolu_${index}`;
        const name = toolIndexes.has(index) ? undefined : '';
        yield openAIChunk(model, { tool_calls: [{ index, id, type: 'function', function: { ...(name !== undefined ? { name } : {}), arguments: data.delta.partial_json } }] }, null);
        if (!toolIndexes.has(index)) toolIndexes.set(index, id);
      }
      return;
    }
    if (event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = typeof data.index === 'number' ? data.index : 0;
      const id = typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${index}`;
      toolIndexes.set(index, id);
      yield openAIChunk(model, { tool_calls: [{ index, id, type: 'function', function: { name: typeof data.content_block.name === 'string' ? data.content_block.name : '', arguments: '' } }] }, null);
      return;
    }
    if (event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') finish = anthropicStopToOpenAI(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
      return;
    }
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const chunk of frameEvents(frame)) yield chunk;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const chunk of frameEvents(buffer)) yield chunk;
  } finally {
    reader.releaseLock();
  }
  if (!sawFinish) {
    sawFinish = true;
    yield openAIChunk(model, {}, finish ?? 'stop');
  }
  yield openAIUsageChunk(model, inputTokens, outputTokens);
  yield 'data: [DONE]\n\n';
}

/** An OpenAI image_url part as a Gemini content part: inline data or a file URI. */
function geminiImageOf(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  return { fileData: { fileUri: url } };
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. The
 * model is not in the body: it goes in the request path. System messages
 * become systemInstruction, tool messages become functionResponse parts, and
 * max_tokens, temperature, top_p, stop sequences and JSON mode land in
 * generationConfig.
 */
export function openAIToGemini(body: Json): Json {
  const systemParts: Json[] = [];
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push({ text });
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          const name = typeof call.function.name === 'string' ? call.function.name : '';
          if (typeof call.id === 'string' && name !== '') toolNames.set(call.id, name);
          let args: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) args = parsed;
            } catch {
              // Not JSON: an empty input.
            }
          }
          parts.push({ functionCall: { name, args } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const name = typeof raw.tool_call_id === 'string' ? toolNames.get(raw.tool_call_id) ?? 'tool' : 'tool';
      const text = typeof content === 'string' ? content : textOf(content);
      contents.push({ role: 'user', parts: [{ functionResponse: { name, response: { result: text } } }] });
      continue;
    }
    // user (and anything else): text and image parts.
    if (typeof content === 'string') {
      if (content !== '') contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
      const image = geminiImageOf(part);
      if (image !== null) parts.push(image);
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = numberOrZero(body.max_tokens) || numberOrZero(body.max_completion_tokens);
  if (maxTokens > 0) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stringListOf(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** A Gemini finishReason as an OpenAI finish_reason. */
function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** The text of a Gemini content's parts, joined the way the driver's parser does. */
function geminiTextOf(content: unknown): string {
  if (!isRecord(content) || !Array.isArray(content.parts)) return '';
  return content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Chat Completions answer object from a Gemini generateContent JSON body. */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const text = geminiTextOf(content);
  const toolCalls: Json[] = [];
  if (Array.isArray(content.parts)) {
    for (const part of content.parts) {
      if (!isRecord(part) || !isRecord(part.functionCall)) continue;
      const call = part.functionCall;
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: typeof call.name === 'string' ? call.name : '',
          arguments: JSON.stringify(isRecord(call.args) ? call.args : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: `chatcmpl-${model}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: geminiStopToOpenAI(first.finishReason),
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

/**
 * A Gemini :streamGenerateContent SSE stream as OpenAI chat-completion chunks:
 * text parts become content deltas, the frame carrying a finishReason becomes
 * the finish chunk, usageMetadata becomes the usage chunk, and the stream ends
 * with [DONE].
 */
export async function* geminiEventsToOpenAIChunks(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let finish: string | null = null;
  let sawFinish = false;

  const frameEvents = function* (frame: string): Generator<string> {
    let dataText = '';
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) dataText += `${line.slice(5).replace(/^ /, '')}\n`;
    }
    if (dataText.trim() === '') return;
    let data: unknown;
    try {
      data = JSON.parse(dataText);
    } catch {
      return;
    }
    if (!isRecord(data)) return;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : null;
    if (first !== null) {
      const text = geminiTextOf(first.content);
      if (text !== '') yield openAIChunk(model, { content: text }, null);
      if (typeof first.finishReason === 'string') finish = geminiStopToOpenAI(first.finishReason);
    }
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const chunk of frameEvents(frame)) yield chunk;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const chunk of frameEvents(buffer)) yield chunk;
  } finally {
    reader.releaseLock();
  }
  if (!sawFinish) {
    sawFinish = true;
    yield openAIChunk(model, {}, finish ?? 'stop');
  }
  yield openAIUsageChunk(model, inputTokens, outputTokens);
  yield 'data: [DONE]\n\n';
}

/**
 * Turn an async iterable of OpenAI SSE chunk strings into a ReadableStream, so
 * the starter's Anthropic event translator can consume a translated stream.
 */
export function chunksToStream(chunks: AsyncIterable<string>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const iterator = chunks[Symbol.asyncIterator]();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const next = await iterator.next();
      if (next.done) {
        controller.close();
        return;
      }
      controller.enqueue(encoder.encode(next.value));
    },
    async cancel() {
      await iterator.return?.();
    },
  });
}
