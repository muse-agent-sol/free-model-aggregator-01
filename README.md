# Free Model Aggregator

One server that speaks the OpenAI Chat Completions API and the Anthropic
Messages API (JSON and streaming) and answers every request with the best
free model it can get from ten free-tier providers.

For each request it reads the brief's quality table and workload manifest,
ranks every free model that meets the request's requirements (tools, vision,
JSON mode, context) by rating, prefers the provider whose wire matches the
client's, and fails over to the next candidate whenever a provider refuses.
Requests and answers are translated between the OpenAI, Anthropic and Gemini
wires — system prompt, user text, images, tool definitions and results, stop
sequences, the token cap and JSON mode included — and each translation is
verified field by field before it is sent, so nothing the client sent is
dropped or rewritten on the way.

Run it with `--list` to print the catalog it registers; run it plainly to
serve a trial. Node 22 standard library only (TypeScript, no build step).
