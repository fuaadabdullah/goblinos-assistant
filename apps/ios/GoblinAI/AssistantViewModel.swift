import Foundation

@MainActor
final class AssistantViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = [
        ChatMessage(role: .assistant, text: "I’m online. What are we handling?")
    ]
    @Published var composerText = ""
    @Published var isThinking = false
    @Published var activeRoute = "Auto"
    @Published var memoryEnabled = true
    @Published var connectionLabel = "LOCAL PREVIEW"

    let quickActions: [QuickAction] = [
        .init(icon: "calendar", title: "Today", subtitle: "Schedule + priorities", prompt: "What should I know about today?"),
        .init(icon: "brain.head.profile", title: "Think", subtitle: "Reason this through", prompt: "Help me think through something."),
        .init(icon: "terminal", title: "Build", subtitle: "Code + systems", prompt: "Help me work on my code."),
        .init(icon: "sparkles", title: "Create", subtitle: "Turn ideas into output", prompt: "Help me create something.")
    ]

    private let api = GoblinAPIClient()
    private var conversationId: String?

    func useQuickAction(_ action: QuickAction) {
        composerText = action.prompt
    }

    func sendCurrentMessage() {
        let clean = composerText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty, !isThinking else { return }

        composerText = ""
        messages.append(.init(role: .user, text: clean))
        isThinking = true

        Task {
            do {
                // Flip this on after setting GoblinAPIClient.baseURL.
                let useLiveBackend = false

                if useLiveBackend {
                    let response = try await api.send(
                        message: clean,
                        conversationId: conversationId,
                        route: activeRoute,
                        memoryEnabled: memoryEnabled
                    )
                    messages.append(.init(role: .assistant, text: response.message))
                    connectionLabel = response.model?.uppercased() ?? "GOBLINOS"
                } else {
                    try? await Task.sleep(for: .milliseconds(650))
                    messages.append(.init(role: .assistant, text: mockReply(for: clean)))
                }
            } catch {
                messages.append(.init(role: .assistant, text: "Backend connection failed. Check the GoblinOS URL and endpoint configuration."))
                connectionLabel = "OFFLINE"
            }
            isThinking = false
        }
    }

    private func mockReply(for message: String) -> String {
        let lower = message.lowercased()
        if lower.contains("today") || lower.contains("schedule") {
            return "Calendar context is ready for wiring. Once connected, I’ll collapse classes, meetings, reminders, traffic, and messages into one briefing."
        }
        if lower.contains("code") || lower.contains("build") {
            return "Build mode ready. The iPhone client can route coding work through GoblinOS and hand heavier inference to your local node."
        }
        return "Interface is live in preview mode. Connect GoblinOS and this becomes the mobile shell for your memory, models, and tools."
    }
}
