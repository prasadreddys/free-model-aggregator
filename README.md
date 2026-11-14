# Free Model Aggregator — make.fun entry

One server that answers OpenAI Chat Completions and Anthropic Messages requests
with the best free model it can get, across ten simulated free-tier providers.

## How it works

- Probes every provider's catalog route in the provider's own wire and builds
  one catalog (the `--list` document and the register body, byte for byte).
- Ranks candidates by the sealed quality table's rating, with a 6-milli spread
  penalty per call already made to a provider, so the top tier (within 17
  milli) spreads across providers instead of burning one budget out.
- On a refusal it reads the error's message: "budget is spent" or "is out"
  exhausts the provider for the trial; "refused this call" is the provider's
  failure schedule and retries the same candidate once; anything else moves to
  the next best legal candidate. Exactly one provider call answers a request.
- Translates all six client-x-provider wire combinations, JSON and SSE. The
  provider's answer text passes through verbatim, and streamed OpenAI calls
  ask the provider for `include_usage`, so the driver's echo, stop-reason and
  usage checks all have their numbers.
- JSON-mode requests never go to an Anthropic-wire provider, whose wire
  cannot carry JSON mode.

## Run

```sh
node main.ts --list        # print the catalog and exit
node main.ts               # serve; needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE
```

Node 22 standard library only; no dependencies. `lib/providers.ts` and
`lib/wire.ts` are the kit starter's library files; the routing, failover and
the Anthropic/Gemini translators (`lib/upstream.ts`) are this entry's own.

## License

MIT. See LICENSE.
