import SwiftUI

@main
struct GoblinAIApp: App {
    @StateObject private var assistant = AssistantViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(assistant)
                .preferredColorScheme(.dark)
        }
    }
}
