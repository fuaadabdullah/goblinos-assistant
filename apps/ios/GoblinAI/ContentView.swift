import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var viewModel: AssistantViewModel
    @State private var showMemory = false
    @State private var showRoutes = false
    @State private var showAttachmentMenu = false

    private let background = Color(red: 0.025, green: 0.032, blue: 0.028)
    private let panel = Color.white.opacity(0.055)
    private let stroke = Color.white.opacity(0.085)

    var body: some View {
        ZStack {
            background.ignoresSafeArea()
            ambientGlow

            VStack(spacing: 0) {
                header
                conversation
                composer
            }
        }
        .sheet(isPresented: $showMemory) {
            MemorySheet(memoryEnabled: $viewModel.memoryEnabled)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $showRoutes) {
            RouteSheet(activeRoute: $viewModel.activeRoute)
                .presentationDetents([.medium])
                .presentationDragIndicator(.visible)
        }
        .confirmationDialog("Add context", isPresented: $showAttachmentMenu) {
            Button("Photo") {}
            Button("Camera") {}
            Button("File") {}
            Button("Cancel", role: .cancel) {}
        }
    }

    private var ambientGlow: some View {
        GeometryReader { geo in
            Circle()
                .fill(RadialGradient(colors: [Color.green.opacity(0.18), Color.green.opacity(0.045), .clear], center: .center, startRadius: 5, endRadius: 180))
                .frame(width: 380, height: 380)
                .blur(radius: 34)
                .offset(x: geo.size.width * 0.40, y: -190)
        }
        .allowsHitTesting(false)
    }

    private var header: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 14).fill(Color.green.opacity(0.13)).frame(width: 44, height: 44)
                Image(systemName: "waveform.path.ecg.rectangle").font(.system(size: 19, weight: .semibold)).foregroundStyle(.green)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("GOBLIN").font(.system(size: 17, weight: .black, design: .rounded)).tracking(1.5)
                HStack(spacing: 5) {
                    Circle().fill(.green).frame(width: 6, height: 6)
                    Text(viewModel.connectionLabel).font(.system(size: 9, weight: .bold, design: .monospaced)).foregroundStyle(.secondary)
                }
            }

            Spacer()
            headerButton("brain") { showMemory = true }
            headerButton("slider.horizontal.3") { showRoutes = true }
        }
        .padding(.horizontal, 18)
        .padding(.top, 10)
        .padding(.bottom, 13)
    }

    private func headerButton(_ icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 17, weight: .semibold))
                .frame(width: 38, height: 38)
                .background(panel, in: Circle())
                .overlay(Circle().stroke(stroke, lineWidth: 1))
        }.buttonStyle(.plain)
    }

    private var conversation: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 16) {
                    hero
                    ForEach(viewModel.messages) { message in MessageBubble(message: message).id(message.id) }
                    if viewModel.isThinking { ThinkingBubble().id("thinking") }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 18)
            }
            .scrollIndicators(.hidden)
            .onChange(of: viewModel.messages.count) { _, _ in scrollToBottom(proxy) }
            .onChange(of: viewModel.isThinking) { _, _ in scrollToBottom(proxy) }
        }
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.24)) {
            if viewModel.isThinking { proxy.scrollTo("thinking", anchor: .bottom) }
            else if let last = viewModel.messages.last { proxy.scrollTo(last.id, anchor: .bottom) }
        }
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 6) {
                Text("What do you need?").font(.system(size: 29, weight: .bold, design: .rounded))
                Text("Your context. Your tools. Your models.").font(.system(size: 14, weight: .medium)).foregroundStyle(.secondary)
            }

            LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 10) {
                ForEach(viewModel.quickActions) { action in
                    Button { viewModel.useQuickAction(action) } label: {
                        VStack(alignment: .leading, spacing: 9) {
                            Image(systemName: action.icon).font(.system(size: 15, weight: .semibold)).foregroundStyle(.green)
                                .frame(width: 30, height: 30).background(Color.green.opacity(0.1), in: RoundedRectangle(cornerRadius: 9))
                            Text(action.title).font(.system(size: 14, weight: .semibold))
                            Text(action.subtitle).font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(1)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(13)
                        .background(panel, in: RoundedRectangle(cornerRadius: 16))
                        .overlay(RoundedRectangle(cornerRadius: 16).stroke(stroke, lineWidth: 1))
                    }.buttonStyle(.plain)
                }
            }
        }
        .padding(.top, 10)
        .padding(.bottom, 6)
    }

    private var composer: some View {
        VStack(spacing: 9) {
            HStack(spacing: 8) {
                Button { showRoutes = true } label: {
                    HStack(spacing: 5) { Image(systemName: "point.3.connected.trianglepath.dotted"); Text(viewModel.activeRoute.uppercased()) }
                        .font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundStyle(.secondary)
                }.buttonStyle(.plain)
                Spacer()
                HStack(spacing: 5) { Image(systemName: viewModel.memoryEnabled ? "memorychip.fill" : "memorychip"); Text(viewModel.memoryEnabled ? "MEMORY ON" : "MEMORY OFF") }
                    .font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundStyle(viewModel.memoryEnabled ? .green : .secondary)
            }.padding(.horizontal, 6)

            HStack(alignment: .bottom, spacing: 10) {
                Button { showAttachmentMenu = true } label: {
                    Image(systemName: "plus").font(.system(size: 18, weight: .semibold)).frame(width: 40, height: 40).background(Color.white.opacity(0.07), in: Circle())
                }.buttonStyle(.plain)

                TextField("Ask anything", text: $viewModel.composerText, axis: .vertical)
                    .lineLimit(1...5).textFieldStyle(.plain).font(.system(size: 16)).padding(.vertical, 10)

                if viewModel.composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Button {} label: { Image(systemName: "waveform").font(.system(size: 18, weight: .semibold)).frame(width: 40, height: 40) }.buttonStyle(.plain)
                } else {
                    Button { viewModel.sendCurrentMessage() } label: {
                        Image(systemName: "arrow.up").font(.system(size: 17, weight: .bold)).foregroundStyle(.black).frame(width: 40, height: 40).background(.green, in: Circle())
                    }.buttonStyle(.plain).disabled(viewModel.isThinking)
                }
            }
            .padding(8)
            .background(RoundedRectangle(cornerRadius: 24).fill(Color(red: 0.07, green: 0.085, blue: 0.075)))
            .overlay(RoundedRectangle(cornerRadius: 24).stroke(stroke, lineWidth: 1))
        }
        .padding(.horizontal, 12).padding(.top, 10).padding(.bottom, 8)
        .background(.ultraThinMaterial.opacity(0.95))
    }
}

private struct MessageBubble: View {
    let message: ChatMessage
    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            if message.role == .user { Spacer(minLength: 48) }
            if message.role == .assistant {
                Image(systemName: "waveform.path.ecg").font(.system(size: 11, weight: .bold)).foregroundStyle(.green)
                    .frame(width: 28, height: 28).background(Color.green.opacity(0.1), in: Circle())
            }
            Text(message.text).font(.system(size: 15.5)).textSelection(.enabled).padding(.horizontal, 14).padding(.vertical, 11)
                .background(message.role == .user ? Color.green.opacity(0.14) : Color.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 18))
                .overlay(RoundedRectangle(cornerRadius: 18).stroke(message.role == .user ? Color.green.opacity(0.18) : Color.white.opacity(0.07), lineWidth: 1))
            if message.role == .assistant { Spacer(minLength: 30) }
        }
    }
}

private struct ThinkingBubble: View {
    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "waveform.path.ecg").font(.system(size: 11, weight: .bold)).foregroundStyle(.green).frame(width: 28, height: 28).background(Color.green.opacity(0.1), in: Circle())
            HStack(spacing: 7) { ProgressView().controlSize(.small); Text("ROUTING").font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundStyle(.secondary) }
                .padding(.horizontal, 13).padding(.vertical, 10).background(Color.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 18))
            Spacer()
        }
    }
}

private struct MemorySheet: View {
    @Binding var memoryEnabled: Bool
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            List {
                Section { Toggle("Use personal memory", isOn: $memoryEnabled) } footer: { Text("Stored context can be included when handling your requests.") }
                Section("Context Layers") {
                    Label("Profile", systemImage: "person.crop.circle")
                    Label("Projects", systemImage: "hammer")
                    Label("Preferences", systemImage: "heart.text.square")
                    Label("Recent activity", systemImage: "clock.arrow.circlepath")
                }
            }
            .navigationTitle("Memory")
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() } } }
        }
    }
}

private struct RouteSheet: View {
    @Binding var activeRoute: String
    @Environment(\.dismiss) private var dismiss
    private let routes = [("Auto", "point.3.connected.trianglepath.dotted", "Let Goblin choose"), ("Fast", "bolt.fill", "Lowest latency"), ("Deep", "brain.head.profile.fill", "Higher reasoning"), ("Local", "desktopcomputer", "Use your own node")]
    var body: some View {
        NavigationStack {
            List {
                ForEach(routes, id: \.0) { route in
                    Button { activeRoute = route.0; dismiss() } label: {
                        HStack(spacing: 12) {
                            Image(systemName: route.1).frame(width: 24)
                            VStack(alignment: .leading, spacing: 2) { Text(route.0).fontWeight(.semibold).foregroundStyle(.primary); Text(route.2).font(.caption).foregroundStyle(.secondary) }
                            Spacer()
                            if activeRoute == route.0 { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                        }
                    }
                }
            }
            .navigationTitle("Routing")
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() } } }
        }
    }
}

#Preview {
    ContentView().environmentObject(AssistantViewModel()).preferredColorScheme(.dark)
}
