//
//  Sidebar.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 19/3/23.
//

import Cocoa
import SwiftUI

enum Panel: Hashable {
    case model
    case controlNet
    case log
}

struct Sidebar: View {
    @Binding var path: NavigationPath
    @Binding var selection: Panel
    @State var showUpdateButton: Bool = false
    
    var body: some View {
        List(selection: $selection) {
            NavigationLink(value: Panel.model) {
                Label("Model", systemImage: "shippingbox")
            }
            NavigationLink(value: Panel.controlNet) {
                Label("ControlNet", systemImage: "cube.transparent")
            }
            NavigationLink(value: Panel.log) {
                Label("Log", systemImage: "doc.text.below.ecg")
            }
        }
        .safeAreaInset(edge: .bottom, content: {
            VStack(spacing: 12) {
                if showUpdateButton {
                    Button {
                        NSWorkspace.shared.open(URL(string: "https://huggingface.co/Guernika/CoreMLStableDiffusion/blob/main/GuernikaModelConverter.dmg")!)
                    } label: {
                        Text("Update available")
                            .frame(minWidth: 168)
                    }.controlSize(.large)
                        .buttonStyle(.borderedProminent)
                }
                if !FileManager.default.fileExists(atPath: "/Applications/Guernika.app") {
                    Button {
                        NSWorkspace.shared.open(URL(string: "macappstore://apps.apple.com/app/id1660407508")!)
                    } label: {
                        Text("Install Guernika")
                            .frame(minWidth: 168)
                    }.controlSize(.large)
                }
            }
            .padding(16)
        })
        .navigationTitle("Guernika Model Converter")
        .navigationSplitViewColumnWidth(min: 200, ideal: 200)
        .onAppear { checkForUpdate() }
    }
    
    func checkForUpdate() {
        Task.detached {
            let versionUrl = URL(string: "https://huggingface.co/Guernika/CoreMLStableDiffusion/raw/main/GuernikaModelConverter.version")!
            guard let lastVersionString = try? String(contentsOf: versionUrl) else { return }
            let lastVersion = Version(stringLiteral: lastVersionString)
            let currentVersionString = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? ""
            let currentVersion = Version(stringLiteral: currentVersionString)
            await MainActor.run {
                withAnimation {
                    showUpdateButton = currentVersion < lastVersion
                }
            }
        }
    }
}

struct Sidebar_Previews: PreviewProvider {
    struct Preview: View {
        @State private var selection: Panel = Panel.model
        var body: some View {
            Sidebar(path: .constant(NavigationPath()), selection: $selection)
        }
    }
    
    static var previews: some View {
        NavigationSplitView {
            Preview()
        } detail: {
           Text("Detail!")
        }
    }
}
