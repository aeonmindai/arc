//! Swift trajectories (4).

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

pub fn all() -> Vec<Trajectory> {
    vec![
        sw_038_swiftui_list(),
        sw_039_async_url(),
        sw_040_combine_to_async(),
        sw_041_codable_strategy(),
    ]
}

fn sw_038_swiftui_list() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a SwiftUI ProjectsListView in Views/ProjectsListView.swift that renders an [Project] from a @StateObject ProjectsViewModel with .listStyle(.insetGrouped). Pull-to-refresh should call viewModel.reload().",
    ));
    turns.push(assistant_call(
        "Reading the view model.",
        "call_001",
        "read_file",
        json!({"path":"ViewModels/ProjectsViewModel.swift"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import Foundation\n\n@MainActor\nfinal class ProjectsViewModel: ObservableObject {\n    @Published var projects: [Project] = []\n    func reload() async { /* fetch */ }\n}\n",
    ));
    turns.push(assistant_call(
        "Writing the view.",
        "call_002",
        "write_file",
        json!({
            "path":"Views/ProjectsListView.swift",
            "contents":"import SwiftUI\n\nstruct ProjectsListView: View {\n    @StateObject private var viewModel = ProjectsViewModel()\n\n    var body: some View {\n        List(viewModel.projects) { project in\n            NavigationLink(project.name) { ProjectDetailView(id: project.id) }\n        }\n        .listStyle(.insetGrouped)\n        .refreshable { await viewModel.reload() }\n        .task { await viewModel.reload() }\n        .navigationTitle(\"Projects\")\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "wrote 478 bytes to Views/ProjectsListView.swift",
    ));
    turns.push(assistant_text(
        "View loads on appear via .task and supports pull-to-refresh through .refreshable.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_038",
        "swift",
        src("github.com/apple/swift", "Apache-2.0", "swift-5.10"),
        turns,
    )
}

fn sw_039_async_url() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Refactor APIClient.fetchProjects from URLSession completion handlers to async/await using URLSession.shared.data(for:). Return [Project] or throw.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"Services/APIClient.swift"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import Foundation\n\nfinal class APIClient {\n    func fetchProjects(completion: @escaping (Result<[Project], Error>) -> Void) {\n        let url = URL(string: \"https://api.example.com/projects\")!\n        URLSession.shared.dataTask(with: url) { data, _, err in\n            if let err = err { completion(.failure(err)); return }\n            guard let data = data else { completion(.failure(URLError(.badServerResponse))); return }\n            do {\n                let out = try JSONDecoder().decode([Project].self, from: data)\n                completion(.success(out))\n            } catch { completion(.failure(error)) }\n        }.resume()\n    }\n}\n",
    ));
    turns.push(assistant_call(
        "Refactoring.",
        "call_002",
        "write_file",
        json!({
            "path":"Services/APIClient.swift",
            "contents":"import Foundation\n\nfinal class APIClient {\n    func fetchProjects() async throws -> [Project] {\n        let url = URL(string: \"https://api.example.com/projects\")!\n        let (data, response) = try await URLSession.shared.data(for: URLRequest(url: url))\n        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {\n            throw URLError(.badServerResponse)\n        }\n        return try JSONDecoder().decode([Project].self, from: data)\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "wrote 540 bytes to Services/APIClient.swift",
    ));
    turns.push(assistant_text(
        "Async version is half the lines and surfaces non-2xx as URLError; call sites need to migrate to `try await client.fetchProjects()`.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_039",
        "swift",
        src("github.com/Alamofire/Alamofire", "MIT", "5.9"),
        turns,
    )
}

fn sw_040_combine_to_async() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Rewrite LocationStore.observe(_:) — currently a Combine AnyPublisher — to use an AsyncStream<Location>. Keep the same source (CoreLocation manager).",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"Stores/LocationStore.swift"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import Combine\nimport CoreLocation\n\nfinal class LocationStore: NSObject, CLLocationManagerDelegate {\n    private let manager = CLLocationManager()\n    private let subject = PassthroughSubject<Location, Never>()\n    func observe() -> AnyPublisher<Location, Never> { subject.eraseToAnyPublisher() }\n    func locationManager(_ m: CLLocationManager, didUpdateLocations locs: [CLLocation]) {\n        guard let l = locs.last else { return }\n        subject.send(Location(lat: l.coordinate.latitude, lon: l.coordinate.longitude))\n    }\n}\n",
    ));
    turns.push(assistant_call(
        "Rewriting with AsyncStream.",
        "call_002",
        "write_file",
        json!({
            "path":"Stores/LocationStore.swift",
            "contents":"import CoreLocation\n\nfinal class LocationStore: NSObject, CLLocationManagerDelegate {\n    private let manager = CLLocationManager()\n    private var continuation: AsyncStream<Location>.Continuation?\n\n    func observe() -> AsyncStream<Location> {\n        AsyncStream { cont in\n            self.continuation = cont\n            cont.onTermination = { @Sendable _ in self.continuation = nil }\n        }\n    }\n\n    func locationManager(_ m: CLLocationManager, didUpdateLocations locs: [CLLocation]) {\n        guard let l = locs.last else { return }\n        continuation?.yield(Location(lat: l.coordinate.latitude, lon: l.coordinate.longitude))\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "wrote 668 bytes to Stores/LocationStore.swift",
    ));
    turns.push(assistant_text(
        "Combine subject replaced with AsyncStream; consumers iterate `for await loc in store.observe()`.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_040",
        "swift",
        src("github.com/apple/swift", "Apache-2.0", "swift-5.10"),
        turns,
    )
}

fn sw_041_codable_strategy() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "User struct uses default JSONDecoder. Switch to snake_case keys and ISO-8601 dates.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"Models/User.swift"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import Foundation\nstruct User: Codable {\n    let id: Int\n    let firstName: String\n    let createdAt: Date\n}\n",
    ));
    turns.push(assistant_call(
        "Reading the decoder factory.",
        "call_002",
        "read_file",
        json!({"path":"Networking/Decoder.swift"}),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "import Foundation\nenum DecoderFactory { static let json = JSONDecoder() }\n",
    ));
    turns.push(assistant_call(
        "Updating the factory.",
        "call_003",
        "write_file",
        json!({
            "path":"Networking/Decoder.swift",
            "contents":"import Foundation\nenum DecoderFactory {\n    static let json: JSONDecoder = {\n        let d = JSONDecoder()\n        d.keyDecodingStrategy = .convertFromSnakeCase\n        d.dateDecodingStrategy = .iso8601\n        return d\n    }()\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_003",
        "wrote 260 bytes to Networking/Decoder.swift",
    ));
    turns.push(assistant_text(
        "Decoder now converts snake_case keys to camelCase and parses ISO-8601 timestamps. User compiles as-is.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_041",
        "swift",
        src("github.com/apple/swift-foundation", "Apache-2.0", "0.0.5"),
        turns,
    )
}
