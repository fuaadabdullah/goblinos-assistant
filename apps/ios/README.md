# Goblin iOS

Native SwiftUI mobile shell for GoblinOS.

## Current interface
- Chat composer and assistant messages
- Auto / Fast / Deep / Local routing modes
- Personal memory toggle and context sheet
- Quick actions for Today, Think, Build, and Create
- Attachment menu and voice-control surface
- GoblinOS API client scaffold
- Browser preview in `preview/index.html`

## Open in Xcode
1. Create a new iOS App project named `GoblinAI` using SwiftUI.
2. Replace the generated app files with the files in `GoblinAI/`.
3. Target iOS 17 or later.
4. Set the real backend base URL in `GoblinAPIClient.swift`.
5. Set `useLiveBackend = true` in `AssistantViewModel.swift` once the endpoint is ready.

## Proposed backend contract

`POST /api/v1/assistant/chat`

```json
{
  "message": "What's on my schedule today?",
  "conversation_id": null,
  "route": "auto",
  "memory_enabled": true,
  "client": "ios"
}
```

Response:

```json
{
  "message": "You have class at 10:00 AM...",
  "model": "groq/llama",
  "memory_used": true
}
```

## Next native integrations
- Speech framework + AVSpeechSynthesizer
- PhotosPicker + camera capture
- App Intents / Shortcuts / Siri handoff
- Share Extension
- Keychain auth
- Push notifications
- SSE/WebSocket response streaming
