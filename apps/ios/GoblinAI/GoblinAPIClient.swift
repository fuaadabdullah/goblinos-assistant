import Foundation

struct GoblinAPIClient {
    var baseURL: URL

    init(baseURL: URL = URL(string: "https://YOUR-GOBLINOS-HOST")!) {
        self.baseURL = baseURL
    }

    func send(
        message: String,
        conversationId: String?,
        route: String,
        memoryEnabled: Bool
    ) async throws -> AssistantResponse {
        let url = baseURL.appending(path: "/api/v1/assistant/chat")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 45

        request.httpBody = try JSONEncoder().encode(
            AssistantRequest(
                message: message,
                conversationId: conversationId,
                route: route.lowercased(),
                memoryEnabled: memoryEnabled,
                client: "ios"
            )
        )

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }

        return try JSONDecoder().decode(AssistantResponse.self, from: data)
    }
}
