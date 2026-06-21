import SwiftUI

enum InputMode: String, CaseIterable, Identifiable {
    case iPhone = "iPhone Camera"
    case rayBan = "Ray-Ban Camera"

    var id: String { rawValue }
}

struct StreamingDemoView: View {
    @State private var host: String = "192.168.0.119"
    @State private var portText: String = "8000"
    @State private var inputMode: InputMode = .iPhone
    @State private var log: [String] = []

    @StateObject private var manager = RayBanCaptureManager()

    var body: some View {
        NavigationStack {
            Form {
                Section("Laptop Server") {
                    TextField("Laptop IP", text: $host)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    TextField("Port", text: $portText)
                        .keyboardType(.numberPad)
                }

                Section("Input") {
                    Picker("Camera Source", selection: $inputMode) {
                        ForEach(InputMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }

                    Button("Register / Connect Ray-Bans") {
                        manager.registerRayBans()
                        appendLog("Started Ray-Ban registration")
                    }

                    Text("Ray-Ban registration: \(manager.registrationStatus)")
                        .font(.caption)
                    
                    Button("Check Permissions") {
                        manager.debugPermissions()
                    }
                }

                Section("Controls") {
                    HStack {
                        Button(manager.isConnected ? "Disconnect Laptop" : "Connect Laptop") {
                            toggleConnect()
                        }
                        .buttonStyle(.borderedProminent)

                        Button(manager.isStreaming ? "Stop Stream" : "Start Stream") {
                            toggleStreaming()
                        }
                        .buttonStyle(.bordered)
                        .disabled(!manager.isConnected)

                        Button(manager.isTranscribing ? "Stop Transcribing" : "Start Transcribing") {
                            toggleTranscription()
                        }
                        .buttonStyle(.bordered)
                    }

                    Text(manager.isConnected ? "Laptop connected" : "Laptop disconnected")
                        .foregroundStyle(manager.isConnected ? .green : .secondary)
                }

                Section("Live Transcript") {
                    Text(manager.latestTranscript.isEmpty ? "No transcript yet" : manager.latestTranscript)
                }

                Section("Log") {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(log.indices, id: \.self) { i in
                                Text(log[i])
                                    .font(.caption.monospaced())
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(minHeight: 120)
                }
            }
            .navigationTitle("Ray-Ban Bridge")
        }
    }

    private func toggleConnect() {
        if manager.isConnected {
            manager.stop()
            appendLog("Disconnected from laptop")
            return
        }

        guard let port = UInt16(portText) else {
            appendLog("Invalid port")
            return
        }

        Task {
            do {
                try await manager.connect(host: host, port: port)
                appendLog("Connected to laptop \(host):\(port)")
            } catch {
                appendLog("Laptop connect error: \(error.localizedDescription)")
            }
        }
    }

    private func toggleStreaming() {
        if manager.isStreaming {
            manager.stopStreaming()
            appendLog("Stopped streaming")
        } else {
            Task {
                do {
                    try await manager.startStreaming(mode: inputMode)
                    appendLog("Started \(inputMode.rawValue)")
                } catch {
                    appendLog("Start error: \(error.localizedDescription)")
                }
            }
        }
    }

    private func toggleTranscription() {
        if manager.isTranscribing {
            manager.stopTranscription()
            appendLog("Stopped transcription")
        } else {
            manager.startTranscription()
            appendLog("Started transcription")
        }
    }

    private func appendLog(_ message: String) {
        log.append("[\(Date().formatted(date: .omitted, time: .standard))] \(message)")
    }
}

#Preview {
    StreamingDemoView()
}
