import Foundation

struct ChatMessage: Identifiable, Equatable, Codable {
    enum Role: String, Codable {
        case user
        case assistant
        case system
    }

    let id: UUID
    let role: Role
    let text: String
    let timestamp: Date

    init(id: UUID = UUID(), role: Role, text: String, timestamp: Date = Date()) {
        self.id = id
        self.role = role
        self.text = text
        self.timestamp = timestamp
    }
}

struct QuickAction: Identifiable {
    let id = UUID()
    let icon: String
    let title: String
    let subtitle: String
    let prompt: String
}

struct AssistantRequest: Codable {
    let message: String
    let conversationId: String?
    let route: String
    let memoryEnabled: Bool
    let client: String

    enum CodingKeys: String, CodingKey {
        case message
        case conversationId = "conversation_id"
        case route
        case memoryEnabled = "memory_enabled"
        case client
    }
}

struct AssistantResponse: Codable {
    let message: String
    let model: String?
    let memoryUsed: Bool?

    enum CodingKeys: String, CodingKey {
        case message
        case model
        case memoryUsed = "memory_used"
    }
}
